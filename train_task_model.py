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
    logging.info('Loading dataset: {}.'.format(args.train_set))
    dataset_train_1 = NiftyDatasetFromTSV(args.train_set, normalizer=None, aug=args.augmentation)  # Normalization applied later
    dataloader_train_1 = torch.utils.data.DataLoader(dataset_train_1,
                                             shuffle=True,
                                             drop_last=True,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_dataset_workers)
    logging.info('Loaded {} images.'.format(len(dataset_train_1)))

    logging.info('Loading dataset: {}.'.format(args.test_set))
    dataset_val_1 = NiftyDatasetFromTSV(args.test_set, normalizer=None)  # Normalization applied later
    dataloader_val_1 = torch.utils.data.DataLoader(dataset_val_1,
                                             shuffle=False,
                                             drop_last=True,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_dataset_workers)
    logging.info('Loaded {} images.'.format(len(dataset_val_1)))

    printhead('Initializing Model')
    ##### Load the TASK model #####
    if args.model_type == 'regressor':
        task = LeNet3D_regressor(input_size=args.input_shape[0]).to(device)
        get_task_output = get_regressor_output
    if args.model_type == 'classifier':
        task = LeNet3D_classifier(num_classes=args.num_classes).to(device)
        get_task_output = get_classifier_output

    task.train()
    task_parameters = list(task.parameters())
    set_requires_grad(task, True)
    logging.info('Task Model: {}'.format(args.model_type))

    ##### Create OPTIMIZERS #####
    optimizer_task = torch.optim.Adam(filter(lambda p: p.requires_grad, task_parameters), lr=args.learning_rate, betas=(0.5, 0.999))

    ##### Initialize TRAINING Variables #####
    loss_train_discriminator_log = []
    loss_train_task_log = []
    err_train_task_A_log = []
    epoch_times = []
    total_task_loss = 0.0

    ##### Initialize VALIDATION Variables #####
    loss_val_discriminator_log = []
    loss_val_task_log = []
    err_val_task_A_log = []
    val_step = 0
    best_val_error = [0, -1]
    total_discriminator_val_loss, total_task_val_loss = 0.0, 0.0
    val_epochs = []

    printhead('TRAINING LOOP')
    for epoch in range(0, args.epochs):
        epoch_start = time.time()
        error_train_A, num_images_A = 0.0, 0.0

        try: #This try/except catched the KeyboardInterrupt raised by the user and performs clean-up
            for batch_idx, batch_samples in enumerate(tqdm(dataloader_train_1, desc='Epoch {:03d}'.format(epoch), leave=False)):

                A, label_A = batch_samples['image'].to(device), batch_samples[args.label_key].to(device)
                A = args.normalizer(A)
                A_orig = A.clone()

                ##### Get TASK MODEL Outputs #####
                optimizer_task.zero_grad()

                output_task_A_batch, error_A_batch, _, _ = get_task_output(task, A_orig, label_A)

                task_loss = args.task_loss(output_task_A_batch, label_A.float())
                task_loss.backward()
                optimizer_task.step()

                # Populate logs
                error_train_A += error_A_batch
                total_task_loss += task_loss.item()
                num_images_A += A_orig.size(0)

            ### LOG METRICS
            total_task_loss = total_task_loss / num_images_A
            loss_train_task_log.append(total_task_loss)
            error_train_A = error_train_A / num_images_A

            err_train_task_A_log.append(error_train_A)

            ### Get COMPUTATION TIMES
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            epoch_times.append(epoch_duration)
            avg_epoch = torch.mean(torch.as_tensor(epoch_times[-20:])).numpy()
            remaining_time = (config['epochs'] - epoch) * avg_epoch
            remaining_time = time.strftime('%Hh%Mm%Ss', time.gmtime(remaining_time))

            logging.info('TRAIN Epo:{:03d} Loss: {:.4f} Task(A): {:.4f} ETA: {}'
                  .format(epoch, loss_train_task_log[-1], error_train_A, remaining_time))

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
                    error_val_A = 0.0
                    num_images_val_A = 0

                    ### Set ALL MODELS TO EVAL
                    task.eval()

                    for batch_idx, batch_samples in enumerate(tqdm(dataloader_val_1, desc='Val', leave=False)):

                        A, label_A = batch_samples['image'].to(device), batch_samples[args.label_key].to(device)
                        A = args.normalizer(A)
                        A_orig = A.clone()

                        output_task_val_A_batch, error_val_A_batch, _, _ = get_task_output(task, A_orig, label_A)
                        error_val_A += error_val_A_batch

                        task_val_loss = args.task_loss(output_task_val_A_batch, label_A.float())
                        total_task_val_loss += task_val_loss.item()
                        num_images_val_A += A_orig.size(0)

                    ##### Populate Logs
                    total_task_val_loss = total_task_val_loss / num_images_val_A
                    loss_val_task_log.append(total_task_val_loss)
                    error_val_A = error_val_A / num_images_val_A
                    err_val_task_A_log.append(error_val_A)

                    val_epochs.append(epoch)

                    logging.info('VAL Epo:{:3d} Loss: {:.4f} Task(A): {:.4f} Best[Task(A)]: {:.4f}'
                                 .format(epoch, total_task_val_loss, error_val_A, best_val_error[1]))

                    ##### Check for a new best model performance
                    if args.model_type ==  'regressor':
                        better_than_before = 1 if error_val_A < best_val_error[1] or best_val_error[1] == -1 else 0
                    else:
                        better_than_before = 1 if error_val_A > best_val_error[1] or best_val_error[1] == -1 else 0

                    if better_than_before:
                        printhead('NEW BEST VAL Epo {:03d} Prev: {:.4f} New: {:.4f} ...saving model'
                            .format(epoch, best_val_error[1], error_val_A))
                        torch.save(task.state_dict(), os.path.join(args.model_dir, 'val_err_{:.5f}_epoch_{:03d}_{}_A.pt'.format(error_val_A, epoch, args.model_type)))
                        early_stopping_counter = 0
                        best_val_error = [epoch, error_val_A]
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
    torch.save(task.state_dict(), os.path.join(args.model_dir, '{}.pt'.format(args.model_type)))

    printhead('Finished TRAINING.')

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


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Domain Adaptation with Adversarial Training of ISTN')
    parser.add_argument('--dev', default='0', help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--config', default="config/config_task_training.json", help='config file')
    parser.add_argument('--output_dir', default='./task_models', help='output root directory')
    parser.add_argument('--num_dataset_workers', type=int, default=4, help='number of worker to use for each dataset.')
    parser.add_argument('--model_type', required=True, choices={'classifier', 'regressor'}, type=str,
                        help='Type of model: `regressor` or `classifier`.')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    #########################################  OPTIONS  ###################################################
    ### DATASET OPTIONS
    args.train_set = config['train_set']
    assert (os.path.exists(args.train_set)), 'Training data does not exist at {}'.format(args.train_set)

    args.test_set = config['test_set']
    assert (os.path.exists(args.test_set)), 'Validation data does not exist at {}'.format(args.test_set)

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

    ### LOSS FUNCTION OPTIONS
    if args.model_type == 'regressor':
        args.task_loss = torch.nn.MSELoss()
    else:
        args.task_loss = torch.nn.BCELoss()

    ### TRAINING LOOP OPTIONS
    args.early_stopping_epochs = config['early_stopping_epochs']
    args.learning_rate = config['learning_rate']
    args.epochs = config['epochs']
    args.batch_size = config['batch_size']
    args.val_interval = 1 if config['val_interval'] > args.epochs else config['val_interval']

    ### OUTPUT OPTIONS
    args.class_names = '{}'.format(config['dataset_name'])

    args.out = 'DA_{}_{}_{}'.format(args.model_type, args.label_key, args.class_names)

    args.params = '_L_{}_E_{:d}_B_{:d}_nf_{}_A_{}'.format(args.learning_rate, args.epochs, args.batch_size, args.nf, args.augmentation)

    args.out = os.path.join(args.output_dir, args.out + args.params)
    args.model_dir = os.path.join(args.out, 'model')

    os.makedirs(args.model_dir, exist_ok=True)

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
