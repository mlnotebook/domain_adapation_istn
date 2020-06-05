import os
import json
import argparse
import torch
import logging
import time
import numpy as np
import shutil
from tqdm import tqdm
from itertools import cycle

from models.task_models import LeNet3D_regressor, LeNet3D_classifier, Discriminator
from models.itn import ITN3D
from models.stn import BSplineSTN3D, AffineSTN3D

from datasets import NiftyDatasetFromTSV

from utils.image_utils import save_tensor_sample, norm_0_255, nii3d_to_pil2d
from utils.model_utils import initialize_weights, set_requires_grad, get_regressor_output, get_classifier_output
from utils.plotting_utils import plot_grad_flow, plot_metric
from utils.training_utils import printhead, EarlyStop


def train(args, config, remote=False):
    printhead('Starting Training.')

    # Setup devices
    if args.seed: torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print('GPU is not available.')
        return
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")
    logging.info('Using device: {} {}'.format(torch.cuda.get_device_name(device), device))

    # Load the dataset
    printhead('Loading...')
    logging.info('Loading dataset: {}.'.format(args.train_sets[0]))
    dataset_train_1 = NiftyDatasetFromTSV(args.train_sets[0], normalizer=None, aug=args.augmentation)  # Normalization applied later
    dataloader_train_1 = torch.utils.data.DataLoader(dataset_train_1,
                                             shuffle=True,
                                             drop_last=True,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_dataset_workers)
    logging.info('Loaded {} images.'.format(len(dataset_train_1)))

    logging.info('Loading dataset: {}.'.format(args.train_sets[1]))
    dataset_train_2 = NiftyDatasetFromTSV(args.train_sets[1], normalizer=None, aug=args.augmentation)  # Normalization applied later
    dataloader_train_2 = torch.utils.data.DataLoader(dataset_train_2,
                                             shuffle=True,
                                             drop_last=True,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_dataset_workers)
    logging.info('Loaded {} images.'.format(len(dataset_train_2)))

    logging.info('Loading dataset: {}.'.format(args.test_sets[0]))
    dataset_val_1 = NiftyDatasetFromTSV(args.test_sets[0], normalizer=None)  # Normalization applied later
    dataloader_val_1 = torch.utils.data.DataLoader(dataset_val_1,
                                             shuffle=False,
                                             drop_last=True,
                                             batch_size=args.batch_size,
                                             num_workers=0)
    logging.info('Loaded {} images.'.format(len(dataset_val_1)))

    logging.info('Loading dataset: {}.'.format(args.test_sets[1]))
    dataset_val_2 = NiftyDatasetFromTSV(args.test_sets[1], normalizer=None)  # Normalization applied later
    dataloader_val_2 = torch.utils.data.DataLoader(dataset_val_2,
                                             drop_last=True,
                                             shuffle=False,
                                             batch_size=args.batch_size,
                                             num_workers=0)
    logging.info('Loaded {} images.'.format(len(dataset_val_2)))

    printhead('Initializing Models')
    ##### Load the TASK model #####
    if args.model_type == 'regressor':
        task = LeNet3D_regressor(input_size=args.input_shape[0]).to(device)
        get_task_output = get_regressor_output
    if args.model_type == 'classifier':
        task = LeNet3D_classifier(num_classes=args.num_classes).to(device)
        get_task_output = get_classifier_output

    if args.finetune:
        logging.info('Loading {} model: {}'.format(args.model_type, args.task_model[0]))
        logging.info(task.load_state_dict(torch.load(args.task_model[0], map_location=device)))
        logging.info('Model loaded.')
    task.train()
    task_parameters = list(task.parameters())
    set_requires_grad(task, True)
    logging.info('Task Model')

    ##### Load the DISCRIMINATOR model #####
    discriminator = Discriminator().to(device)
    discriminator.apply(initialize_weights)
    discriminator.train()
    discriminator_parameters = list(discriminator.parameters())
    set_requires_grad(discriminator, True)
    logging.info('Discriminator')

    ##### Load the ITN #####
    istn_A2B_parameters = []
    itn_A2B = ITN3D(nf=args.nf).to(device)
    istn_A2B_parameters += list(itn_A2B.parameters())
    itn_A2B.train()
    itn_A2B.apply(initialize_weights)
    set_requires_grad(itn_A2B, True)
    logging.info('ITN')

    ##### Load the STN #####
    stn_A2B = None
    if args.stn:
        assert (args.stn in ['affine', 'bspline']), "STN should be one of `bspline` or `spline`."
        if args.stn == 'bspline':
            stn_A2B = BSplineSTN3D(input_size=args.input_shape[0],
                               device=device,
                               control_point_spacing=(args.cp_spacing[0],
                                                      args.cp_spacing[1],
                                                      args.cp_spacing[2]),
                               max_displacement=args.max_displacement,
                               nf=args.nf).to(device)
        if args.stn == 'affine':
            stn_A2B = AffineSTN3D(input_size=args.input_shape[0],
                                  device=device,
                                  input_channels=1,
                                  nf=args.nf).to(device)
        istn_A2B_parameters += list(stn_A2B.parameters())
        stn_A2B.train()
        # Weights are not initialized. Auto initialized in STN to perform identity transform.
        set_requires_grad(stn_A2B, True)
    logging.info('STN ({})'.format(args.stn))

    ##### Create OPTIMIZERS #####
    optimizer_discriminator = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator_parameters), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_istn = torch.optim.Adam(filter(lambda p: p.requires_grad, istn_A2B_parameters), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_task = torch.optim.Adam(filter(lambda p: p.requires_grad, task_parameters), lr=0.0001)

        
    ##### Initialize TRAINING Variables #####
    loss_train_istn_log = []
    loss_train_discriminator_log = []
    loss_train_task_log = []
    err_train_task_A_log = []
    err_train_task_B_log = []
    err_train_task_A2B_log = []
    early_stopping_counter = 0
    epoch_times = []
    total_istn_A2B_loss = 0.0
    total_discriminator_loss = 0.0
    total_task_loss = 0.0


    ##### Initialize VALIDATION Variables #####
    loss_val_istn_log = []
    loss_val_discriminator_log = []
    loss_val_task_log = []
    err_val_task_A_log = []
    err_val_task_B_log = []
    err_val_task_A2B_log = []
    val_step = 0
    best_val_error = [0, -1]
    error_val_A_base, error_val_B_base = 0.0, 0.0
    num_images_val_A_base, num_images_val_B_base = 0, 0
    total_istn_val_loss, total_discriminator_val_loss, total_task_val_loss = 0.0, 0.0, 0.0
    val_epochs = []

    ##### Create Soft Label Generators
    low = torch.distributions.Uniform(0.00, 0.03)
    high = torch.distributions.Uniform(0.97, 1.00)

    printhead('TRAINING LOOP')
    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        error_train_A2B, error_train_A, error_train_B, num_images_A, num_images_B, num_images_A2B = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Set the ISTN and task model to training mode
        itn_A2B.train()
        set_requires_grad(itn_A2B, True)
        if args.stn:
            stn_A2B.train()
            set_requires_grad(stn_A2B, True)

        task.train()
        set_requires_grad(task, True)

        try: #This try/except catched the KeyboardInterrupt raised by the user and performs clean-up
            for batch_idx, batch_samples in enumerate(zip(tqdm(dataloader_train_1, desc='Epoch {:03d}'.format(epoch), leave=False), cycle(dataloader_train_2))):

                A, label_A = batch_samples[0]['image'].to(device), batch_samples[0][args.label_key].to(device)
                B, label_B = batch_samples[1]['image'].to(device), batch_samples[1][args.label_key].to(device)

                if len(B) != len(A):
                    # If the two datasets are not the same length, correct the size.
                    B, label_B = B[:len(A)], label_B[:len(A)]

                A = args.normalizer(A)
                B = args.normalizer(B)

                # Create clones of the original data for use later
                B_orig = B.clone()
                A_orig = A.clone()

                # GA2B(A) - domain shift
                A2B = itn_A2B(A)
                A2B = stn_A2B(A2B.to(device)) if args.stn else A2B
                # GA2B(B) - identity
                B2B = itn_A2B(B)
                B2B = stn_A2B(B2B.to(device)) if args.stn else B2B

                # Create clones of the original data for use later
                A2B_orig = A2B.clone()

                ##### CALCULATE GENERATOR LOSSES #####
                # GAN loss
                optimizer_istn.zero_grad()
                set_requires_grad(discriminator, False) #discriminator is not being updated here.
                output_Dis_A2B = discriminator(A2B, return_logits=0)
                output_Dis_A2B = output_Dis_A2B.view(A2B.size(0), -1)
                Dis_A2B_High = high.sample(output_Dis_A2B.shape).to(device)
                discriminator_A2B_gan_loss = args.gan_loss(output_Dis_A2B, Dis_A2B_High)
                # Identity loss
                idt_B2B_loss = 0.5 * args.cyc_weight * args.idt_loss(B2B, B) if args.idt_loss is not None else 0
                # Total Loss
                istn_A2B_loss = discriminator_A2B_gan_loss + idt_B2B_loss
                # Perform OPTIMIZER UPDATE
                istn_A2B_loss.backward()
                optimizer_istn.step()

                ##### CALCULATE DISCRIMINATOR LOSSES #####
                optimizer_discriminator.zero_grad()
                set_requires_grad(discriminator, True)
                output_Dis_B = discriminator(B.detach(), return_logits=0)
                output_Dis_B = output_Dis_B.view(B.size(0), -1)
                Dis_B_High = high.sample(output_Dis_B.shape).to(device)
                discriminator_B_loss = args.dis_loss(output_Dis_B, Dis_B_High)

                output_Dis_A2B = discriminator(A2B.detach(), return_logits=0)
                output_Dis_A2B = output_Dis_A2B.view(A2B.size(0), -1)
                Dis_A2B_Low = low.sample(output_Dis_A2B.shape).to(device)
                discriminator_A2B_loss = args.dis_loss(output_Dis_A2B, Dis_A2B_Low)
                #Perform DISCRIMINATOR UPDATE
                discriminator_loss = 0.5 * (discriminator_B_loss + discriminator_A2B_loss)
                discriminator_loss.backward()
                optimizer_discriminator.step()

                ##### Get TASK MODEL Outputs #####
                optimizer_task.zero_grad()
                output_task_A_batch, error_A_batch, _, _ = get_task_output(task, A_orig, label_A)
                output_task_B_batch, error_B_batch, _, _ = get_task_output(task, B_orig, label_B)
                output_task_A2B_batch, error_A2B_batch, _, _ = get_task_output(task, A2B_orig.detach(), label_A)
                # Only update the task model based on the transformed output
                task_loss = args.task_loss(output_task_A2B_batch, label_A.float())
                task_loss.backward()
                optimizer_task.step()

                # Populate logs
                error_train_A += error_A_batch
                error_train_B += error_B_batch
                error_train_A2B += error_A2B_batch

                total_istn_A2B_loss += istn_A2B_loss.item()
                total_discriminator_loss += discriminator_loss.item()
                total_task_loss += task_loss.item()
                
                num_images_A += A_orig.size(0)
                num_images_B += B_orig.size(0)
                num_images_A2B += A2B_orig.size(0)

            ### LOG METRICS
            total_istn_A2B_loss = total_istn_A2B_loss / num_images_A
            total_discriminator_loss = total_discriminator_loss / num_images_A
            total_task_loss = total_task_loss / num_images_A
            
            loss_train_istn_log.append(total_istn_A2B_loss)
            loss_train_discriminator_log.append(total_discriminator_loss)
            loss_train_task_log.append(total_task_loss)

            error_train_A = error_train_A / num_images_A
            error_train_B = error_train_B / num_images_B
            error_train_A2B = error_train_A2B / num_images_A2B

            err_train_task_A_log.append(error_train_A)
            err_train_task_B_log.append(error_train_B)
            err_train_task_A2B_log.append(error_train_A2B)

            ### Get COMPUTATION TIMES
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_times.append(epoch_duration)
            avg_epoch = torch.mean(torch.as_tensor(epoch_times[-20:])).numpy()
            remaining_time = (config['epochs'] - epoch) * avg_epoch
            remaining_time = time.strftime('%Hh%Mm%Ss', time.gmtime(remaining_time))

            logging.info('TRAIN Epo:{:03d} Loss[ISTN/Dis/Tsk]:[{:.3f}/{:.3f}/{:.3f}/] {}[A/A2B/B]:[{:.3f}/{:.3f}/{:.3f}] ETA: {}'
                  .format(epoch,
                  loss_train_istn_log[-1],
                  loss_train_discriminator_log[-1],
                  loss_train_task_log[-1],
                  'MAE' if args.model_type == 'regressor' else 'Acc',
                  error_train_A,
                  error_train_A2B,
                  error_train_B,
                  remaining_time))

        except KeyboardInterrupt:
            printhead('USER TERMINATED at Epoch: {}'.format(epoch))
            if remote:
                raise KeyboardInterrupt
            else:
                break

        ###### VALIDATION STEP ######
        try:
            if (epoch == 0 or epoch % args.val_interval == 0 or epoch == args.epochs-1):
                with torch.no_grad():
                    error_val_A, error_val_B, error_val_A2B = 0.0, 0.0, 0.0
                    num_images_val_A, num_images_val_B, num_images_val_A2B = 0, 0, 0

                    ### Set ALL MODELS TO EVAL
                    discriminator.eval()
                    task.eval()
                    itn_A2B.eval()
                    if args.stn:
                        stn_A2B.eval()

                    for batch_idx, batch_samples in enumerate(zip(tqdm(dataloader_val_1, desc='Val', leave=False), cycle(dataloader_val_2))):

                        A, label_A = batch_samples[0]['image'].to(device), batch_samples[0][args.label_key].to(device)
                        B, label_B = batch_samples[1]['image'].to(device), batch_samples[1][args.label_key].to(device)

                        if len(B) != len(A):
                            # If the two datasets are not the same length, correct the size.
                            B, scanner_B, mask_B, label_B = B[:len(A)], scanner_B[:len(A)], mask_B[:len(A)], label_B[:len(A)]

                        A = args.normalizer(A)
                        B = args.normalizer(B)

                        # Create clones of the original data for use later
                        A_orig = A.clone()
                        B_orig = B.clone()

                        if epoch == 0:
                            # Get baselines on first epoch, and retain them for later
                            output_task_val_A_base_batch, error_val_A_base_batch, _, _ = get_task_output(task, A_orig, label_A)
                            output_task_val_B_base_batch, error_val_B_base_batch, _, _ = get_task_output(task, B_orig, label_B)

                            error_val_A_base += error_val_A_base_batch
                            error_val_B_base += error_val_B_base_batch

                            num_images_val_A_base += A.size(0)
                            num_images_val_B_base += B.size(0)

                        ### Save a sample of validation images on first step
                        if epoch == 0 and batch_idx == 0 and (args.nii or args.png):
                            save_tensor_sample(A, '{}_val_preITN_{}'.format(epoch, 'A'), args.samples_dir, nii=args.nii, png=args.png)

                        A2B = itn_A2B(A)
                        A2B_postITN = A2B.clone()

                        if batch_idx == 0  and (args.nii or args.png):
                            save_tensor_sample(A2B_postITN, '{}_val_postITN_{}'.format(epoch, 'A2B'), args.samples_dir, nii=args.nii, png=args.png)
                            save_tensor_sample(A2B_postITN - A, '{}_val_postITNdiff_{}'.format(epoch, 'A2B'), args.samples_dir, nii=args.nii, png=args.png)

                        A2B = stn_A2B(A2B) if args.stn else A2B
                        A2B_postSTN = A2B.clone()

                        if batch_idx == 0  and (args.nii or args.png):
                            save_tensor_sample(A2B_postSTN, '{}_val_postSTN_{}'.format( epoch, 'A2B'), args.samples_dir, nii=args.nii, png=args.png)
                            save_tensor_sample(A2B_postSTN - A2B_postITN, '{}_val_postSTNdiff_{}'.format( epoch, 'A2B'), args.samples_dir, nii=args.nii, png=args.png)

                        output_task_val_A_batch, error_val_A_batch, _, _ = get_task_output(task, A_orig, label_A)
                        output_task_val_B_batch, error_val_B_batch, _, _ = get_task_output(task, B_orig, label_B)
                        output_task_val_A2B_batch, error_val_A2B_batch, _, _ = get_task_output(task, A2B, label_A)

                        error_val_A += error_val_A_batch
                        error_val_B += error_val_B_batch
                        error_val_A2B += error_val_A2B_batch

                        ##### Get Losses
                        #ISTN
                        output_Dis_val_A2B = discriminator(A2B.detach())
                        output_Dis_val_A2B = output_Dis_val_A2B.view(A2B.size(0), -1)
                        Dis_A2B_High = high.sample(output_Dis_val_A2B.shape).to(device)
                        discriminator_A2B_gan_val_loss = args.gan_loss(output_Dis_val_A2B, Dis_A2B_High)
                        idt_val_B2B_val_loss = 0.5 * args.cyc_weight * args.idt_loss(B2B, B) if args.idt_loss is not None else 0
                        istn_val_loss = discriminator_A2B_gan_val_loss + idt_val_B2B_val_loss
                        #DISCRIMINATOR
                        output_Dis_val_B = discriminator(B.detach())
                        output_Dis_val_B = output_Dis_val_B.view(B.size(0), -1)
                        Dis_B_High = high.sample(output_Dis_val_B.shape).to(device)
                        Dis_A2B_Low = low.sample(output_Dis_val_A2B.shape).to(device)
                        discriminator_B_val_loss = args.dis_loss(output_Dis_val_B, Dis_B_High)
                        discriminator_A2B_val_loss = args.dis_loss(output_Dis_val_A2B, Dis_A2B_Low)
                        discriminator_val_loss = 0.5 * (discriminator_B_val_loss + discriminator_A2B_val_loss)
                        #TASK
                        task_val_loss = args.task_loss(output_task_val_A2B_batch, label_A.float())

                        total_istn_val_loss += istn_val_loss.item()
                        total_discriminator_val_loss += discriminator_val_loss.item()
                        total_task_val_loss += task_val_loss.item()

                        num_images_val_A += A_orig.size(0)
                        num_images_val_B += B_orig.size(0)
                        num_images_val_A2B += A2B.size(0)
                        ## End Batch

                    ##### Populate Logs
                    total_istn_val_loss = total_istn_val_loss / num_images_val_A
                    total_discriminator_val_loss = total_discriminator_val_loss / num_images_val_A
                    total_task_val_loss = total_task_val_loss / num_images_val_A

                    loss_val_istn_log.append(total_istn_val_loss)
                    loss_val_discriminator_log.append(total_discriminator_val_loss)
                    loss_val_task_log.append(total_task_val_loss)

                    error_val_A = error_val_A / num_images_val_A
                    error_val_B = error_val_B / num_images_val_B
                    error_val_A2B = error_val_A2B / num_images_val_A2B

                    err_val_task_A_log.append(error_val_A)
                    err_val_task_B_log.append(error_val_B)
                    err_val_task_A2B_log.append(error_val_A2B)

                    val_epochs.append(epoch)

                    if epoch == 0:
                        # Get baselines on first epoch, and retain them
                        error_val_A_base = error_val_A_base / num_images_val_A_base
                        error_val_B_base = error_val_B_base / num_images_val_B_base

                    logging.info('VAL Epo:{:3d} {}[A0/A/A2B]:[{:.4f}/{:.4f}/{:.4f}] {}[B0/B/Del]:[{:.4f}/{:.4f}/{:.4f}] Best[B]:[{:.4f}]'
                                 .format(epoch, 'MAE' if args.model_type=='regressor' else 'Acc',
                                         error_val_A_base, error_val_A, error_val_A2B,
                                         'MAE' if args.model_type == 'regressor' else 'Acc',
                                         error_val_B_base, error_val_B, np.abs(error_val_B_base - error_val_B),
                                         best_val_error[1]))

                    ##### Check for a new best model performance
                    if args.model_type ==  'regressor':
                        better_than_before = 1 if error_val_A2B < best_val_error[1] or best_val_error[1] == -1 else 0
                    else:
                        better_than_before = 1 if error_val_A2B > best_val_error[1] or best_val_error[1] == -1 else 0

                    if better_than_before:
                        printhead('NEW BEST VAL A2B:{:3d} Prev [{:.5f}] New [{:.5f}] ...saving model'
                            .format(epoch, best_val_error[1], error_val_A2B))
                        torch.save(itn_A2B.state_dict(), os.path.join(args.model_dir,'val_err_{:.5f}_epoch_{:03d}_itn_A2B.pt'.format(error_val_A2B, epoch)))
                        if args.stn:
                            torch.save(stn_A2B.state_dict(), os.path.join(args.model_dir,'val_err_{:.5f}_epoch_{:03d}_stn_A2B.pt'.format(error_val_A2B, epoch)))
                        torch.save(task.state_dict(), os.path.join(args.model_dir, 'val_err_{:.5f}_epoch_{:03d}_{}_A2B.pt'.format(error_val_A2B, epoch, args.model_type)))
                        early_stopping_counter = 0
                        best_val_error = [epoch, error_val_A2B]
                    else:
                        early_stopping_counter += 1

                    ##### Check number of validation steps - must have trained for at least 5 epochs.
                    val_step += 1
                    if val_step > 5 and early_stopping_counter == args.early_stopping_epochs:
                        printhead('EARLY STOPPING TRIGGER: No change in val_accuracy for {} epochs'.format(
                            early_stopping_counter))
                        raise EarlyStop
        except EarlyStop:
            break

    ##### After the training loop - save the final models
    ### NOTE: May not be the best models - this just saves the final epoch step just in case it's needed.
    torch.save(itn_A2B.state_dict(), os.path.join(args.model_dir, 'itn.pt'))
    if args.stn:
        torch.save(stn_A2B.state_dict(), os.path.join(args.model_dir, 'stn.pt'))
    torch.save(task.state_dict(), os.path.join(args.model_dir, '{}.pt'.format(args.model_type)))

    printhead('Finished TRAINING.')

    plot_metric({'ISTN Train': [loss_train_istn_log, range(len(loss_train_istn_log))],
               'ISTN Val': [loss_val_istn_log, val_epochs]},
              'ISTN Losses', 'Loss', args)
    plot_metric({'Dis Train': [loss_train_discriminator_log, range(len(loss_train_discriminator_log))],
               'Dis Val': [loss_val_discriminator_log, val_epochs]},
              'Discriminator Losses', 'Loss', args)
    plot_metric({'Task Train': [loss_train_task_log, range(len(loss_train_task_log))],
               'Task Val': [loss_val_task_log, val_epochs]},
              'Task Losses', 'Loss', args)

    if args.model_type == 'regressor':
        metric = 'MSE'
    else:
        metric = 'Acc'
        
    plot_metric({'Task(A) Train': [err_train_task_A_log, range(len(err_train_task_A_log))],
               'Task(A) Val': [err_val_task_A_log, val_epochs]},
              'Task(A) {}'.format(metric), metric, args)
    plot_metric({'Task(B) Train': [err_train_task_B_log, range(len(err_train_task_B_log))],
               'Task(B) Val': [err_val_task_B_log, val_epochs]},
              'Task(B) {}'.format(metric), metric, args)
    plot_metric({'Task(A2B) Train': [err_train_task_A2B_log, range(len(err_train_task_A2B_log))],
               'Task(A2B) Val': [err_val_task_A2B_log, val_epochs]},
              'Task(A2B) {}'.format(metric), metric, args)


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Domain Adaptation with Adversarial Training of ISTN')
    parser.add_argument('--dev', default='0', help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--config', default="config/config_train_unidirectional.json", help='config file')
    parser.add_argument('--output_dir', default='./output', help='output root directory')
    parser.add_argument('--num_dataset_workers', type=int, default=4, help='number of worker to use for each dataset.')
    parser.add_argument('--B2A', action='store_true', help='swap siteA and siteB')
    parser.add_argument('--nii', action='store_true', help='save samples as .nii.gz')
    parser.add_argument('--png', action='store_true', help='save samples as .png')
    parser.add_argument('--model_type', required=True, choices={'classifier', 'regressor'}, type=str,
                        help='Type of model: `regressor` or `classifier`.')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    #########################################  OPTIONS  ###################################################
    ### DATASET OPTIONS
    args.train_sets = [config['siteA_train'], config['siteB_train']]
    for i in range(len(args.train_sets)):
        assert (os.path.exists(args.train_sets[i])), 'Training data does not exist at {}'.format(args.train_sets[0])

    args.test_sets = [config['siteA_val'], config['siteB_val']]
    for i in range(len(args.test_sets)):
        assert (os.path.exists(args.test_sets[i])), 'Validation data does not exist at {}'.format(args.test_sets[0])

    args.finetune = config['finetune']
    if args.finetune:
        args.task_model = config['task_model']
        assert(os.path.exists(args.task_model)), 'Finetuning is ON, but task model does not exist at {}'.format(args.task_model)
    else:
        args.task_model = None

    if args.B2A:
        if args.finetune:
            printhead('Note: Both `B2A` & `finetune` passed as args. A & B switched. Check task model is trained on site B.')
        args.training_sets = args.training_sets[::-1]
        args.test_sets = args.test_sets[::-1]

    args.input_shape= config['input_shape']
    args.label_key = config['label_key']

    ### PREPROCESSING OPTIONS
    args.augmentation = config['augmentation']

    assert(config['normalizer'] in ['', 'tanh'])
    if config['normalizer'] == 'tanh':
        args.normalizer = torch.tanh
    else:
        args.normalizer = None

    ### TASK MODEL OPTIONS
    if args.model_type == 'regression':
        args.num_classes = None
    else:
        args.num_classes = config['num_classes']

    ### ISTN OPTIONS
    args.nf = config['nf']
    args.stn = config['stn']
    assert(args.stn in ['none', 'bspline', 'affine']), "STN should be `bspline`, `affine` or `none`."
    if args.stn == 'bspline':
        args.max_displacement = config['max_displacement']
        args.cp_spacing = config['cp_spacing']
    else:
        args.max_displacement = None
        args.cp_spacing = None

    ### LOSS FUNCTION OPTIONS
    loss_functions = {'bce': torch.nn.BCELoss(),
                      'mse': torch.nn.MSELoss(),
                      'l1':  torch.nn.L1Loss()}

    assert (config['gan_loss'] in loss_functions.keys())
    assert (config['idt_loss'] in loss_functions.keys())

    args.gan_loss = loss_functions[config['gan_loss']]
    args.idt_loss = loss_functions[config['idt_loss']]
    args.dis_loss = loss_functions[config['dis_loss']]

    if args.model_type == 'regressor':
        args.task_loss = torch.nn.MSELoss()
    else:
        args.task_loss = torch.nn.BCELoss()

    ### TRAINING LOOP OPTIONS
    args.early_stopping_epochs = config['early_stopping_epochs']
    args.learning_rate = config['learning_rate']
    args.epochs = config['epochs']
    args.batch_size = config['batch_size']
    args.cyc_weight = config['cyc_weight']

    args.val_interval = 1 if config['val_interval'] > args.epochs else config['val_interval']

    ### OUTPUT OPTIONS
    args.class_names = '{}_{}'.format(config['siteA_name'], config['siteB_name'])

    args.out = '{}_{}_STN_{}'.format('UniDA_{}_{}'.format(args.model_type, args.label_key),
                                     args.class_names,
                                     str(args.stn) if args.stn else 'NONE')

    args.params = '_L_{}_E_{:d}_B_{:d}_{}_Cyc_{}_GL_{}_IL_{}_DL_{}_A_{}'.format(args.learning_rate,
                                                                                          args.epochs,
                                                                                          args.batch_size,
                                                                                          '' if args.stn != 'bspline' else 'Sp_{}_MaxD_{}_'.format(args.cp_spacing[0],
                                                                                          args.max_displacement),
                                                                                          args.cyc_weight,
                                                                                          config['gan_loss'],
                                                                                          config['idt_loss'],
                                                                                          config['dis_loss'],
                                                                                          args.augmentation)

    args.out = os.path.join(args.output_dir, args.out + args.params)
    args.model_dir = os.path.join(args.out, 'model')
    args.samples_dir = os.path.join(args.out, 'samples')
    args.code_dir = os.path.join(args.out, 'code')

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.code_dir, exist_ok=True)

    for file in tqdm(config['files'], desc="Copying script files..."):
        shutil.copyfile(file, os.path.join(args.code_dir, os.path.basename(file)))

    ### LOGGING OPTIONS
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-5s %(message)s',
                        datefmt='%d%m %H:%M:%S',
                        filename=os.path.join(args.out, 'log.txt'),
                        filemode='w')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    consoleHandler.setFormatter(formatter)
    logging.getLogger('').addHandler(consoleHandler)

    #################################################################################################################
    train(args, config)
