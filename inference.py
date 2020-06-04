################################################################################
# Applies a trained ISTN to image data and perofms inference of a task model.
# (i.e. regressor or classifier). Plots MAE data for regressor.
#
# R. Robinson June 2020 - Imperial College London - github.com/mlnotebook
################################################################################

import argparse
import json
import os
import logging
import numpy as np
import torch
from tqdm import tqdm

from models.task_models import LeNet3D_regressor, LeNet3D_classifier
from models.itn import ITN3D
from models.stn import BSplineSTN3D, AffineSTN3D
from datasets import NiftyDatasetFromTSV

from utils.model_utils import set_requires_grad, get_regressor_output, get_classifier_output
from utils.plotting_utils import plot_mae, plot_cumulative_threshold
from utils.training_utils import printhead, close_loggers


def infer(args, remote=False):
    """ Performs inference on a PyTorch classification or regression model with or without ISTN.
    Args:
        required:
            args.dev: int - the device to use. If CUDA is not available, CPU is used.
            args.model_type: str - `regressor` or `classifier`, the type of model being evaluated.
            args.test_set: str - path to the .csv file required for the DataHarmonizationDataset loader.
            args.num_dataset_workers: int - the number of threads to spawn for data loading and processing.
            args.input_shape: list of ints - the shape of the input [C, H, W, D].
        oprtional:
            args.seed: int - the seed to be used for random states.
            args.normalizer: function - If not None, the function used to normalize the input to the model.
            args.output_dir: str - path in which to save figures.
            args.itn_path: str - path to the saved itn .pt model.
            args.stn_path: str - path to the saved stn .pt model.
                required if args.stn_path:
                    args.cp_spacing: list of ints - the control point spacing in the STN model.
                    args.max_displacement: float - the maximum displacement for deformations in the STN model.
    Returns:
        output: 2D-Tensor - [B, 1] - the output prediction for each input.
        if model_type == `regression`: average_error: float - the average MAE between input and labels.
        if model_type == `classification`: accuracy: float - the classification accuracy.
    """

    assert (args.model_type in ['regressor', 'classifier']), \
        "`model_type` should be `classifier` or `regressor`."
    assert (os.path.exists(args.task_model)), "Model .pt does not exist at {}".format(args.task_model)
    assert (os.path.exists(args.test_set)), "Dataset does not exist at {}".format(args.test_set)

    printhead('Starting {} Inference.'.format(args.model_type.upper()))
    # Setup devices
    if args.seed: torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        logging.info('GPU is not available.')
        return
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")
    logging.info('Using device: {} {}'.format(torch.cuda.get_device_name(device), device))

    # Load the dataset
    printhead('Loading...')
    logging.info('Loading dataset: {}.'.format(args.test_set))
    dataset = NiftyDatasetFromTSV(args.test_set, normalizer=None, aug=False)  # Normalization applied later
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             drop_last=False,
                                             batch_size=1,
                                             num_workers=2)
    logging.info('Loaded {} images.'.format(len(dataset)))

    # Load the task model
    ##### Load the TASK model #####
    if args.model_type == 'regressor':
        task = LeNet3D_regressor(input_size=args.input_shape[0]).to(device)
        get_task_output = get_regressor_output
    if args.model_type == 'classifier':
        task = LeNet3D_classifier(num_classes=args.num_classes).to(device)
        get_task_output = get_classifier_output

    logging.info('Loading {} model: {}'.format(args.model_type, args.task_model))
    logging.info(task.load_state_dict(torch.load(args.task_model, map_location=device)))
    logging.info('Model loaded.')
    task.eval()
    set_requires_grad(task, False)
    logging.info('Task Model')

    # Load the ITN
    itn = None
    if args.itn and args.itn_path:
        assert (os.path.exists(args.itn_path)), "No ITN .pt file exists at {}".format(args.itn_path)
        itn = ITN3D(nf=args.nf).to(device)
        logging.info('Loading ITN: {}'.format(args.itn_path))
        logging.info(itn.load_state_dict(torch.load(args.itn_path, map_location=device)))
        itn.eval()
        set_requires_grad(itn, False)

    # Load the STN
    stn = None
    if args.stn and args.stn_path:
        assert (os.path.exists(args.stn_path)), "No STN .pt file exists at {}".format(args.stn_path)
        assert (args.stn in ['affine', 'bspline']), "STN should be one of `bspline` or `spline`."
        if args.stn == 'bspline':
            stn = BSplineSTN3D(input_size=args.input_shape[0],
                               device=device,
                               control_point_spacing=(args.cp_spacing[0],
                                                      args.cp_spacing[1],
                                                      args.cp_spacing[2]),
                               max_displacement=args.max_displacement,
                               nf=args.nf).to(device)
        if args.stn == 'affine':
            stn = AffineSTN3D(input_size=args.input_shape[0],
                              device=device,
                              input_channels=1,
                              nf=args.nf).to(device)
        logging.info('Loading STN: {}'.format(args.stn_path))
        logging.info(stn.load_state_dict(torch.load(args.stn_path, map_location=device)))
        stn.eval()
        set_requires_grad(stn, False)

    printhead('Performing INFERENCE...')
    # Perform inference
    dataset_total_err = 0.0
    dataset_abs_errors = []
    dataset_outputs = []
    dataset_labels = []
    dataset_correct_class_results = 0
    dataset_correct_reg_results = np.zeros(len(args.thresholds)).astype(int)

    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(dataloader, desc='Inference')):
            images, labels = batch_samples['image'].to(device), batch_samples[args.label_key].to(device)

            if args.normalizer:
                images = args.normalizer(images)

            if itn:
                images = itn(images)
            if stn:
                images = stn(images)

            if args.model_type == 'regressor':
                batch_output, batch_err, batch_correct_results, abs_errors = get_regressor_output(task, images, labels, thresholds=args.thresholds)

                if labels is not None:
                    dataset_labels += list([labels.item()])
                    dataset_total_err += batch_err
                    dataset_abs_errors += list([abs_errors.item()])
                    dataset_correct_reg_results += np.array(batch_correct_results).astype(int)
                dataset_outputs += list([batch_output.item()])
            else:
                batch_output, batch_correct_results, _, batch_predictions = get_classifier_output(task, images, labels)
                dataset_outputs += list([batch_predictions.item()])
                if labels is not None:
                    dataset_labels += list([labels.item()])
                    dataset_correct_class_results += batch_correct_results

    # Gather results
    if labels is not None:
        if args.model_type == 'regressor':
            dataset_average_err = dataset_total_err / len(dataset)
            printhead('Average MAE = {}'.format(dataset_average_err))

            correctthresh = [float(result) / len(dataset) for result in dataset_correct_reg_results]

            if args.graphs:
                printhead('Generating graphs.')
                # Histogram of absolute errors
                plot_mae(np.array(dataset_abs_errors), output_dir=args.output_dir)
                # Histogram of absolute errors
                plot_cumulative_threshold(correctthresh, args.thresholds, output_dir=args.output_dir)

            if args.display_outputs:
                for sample in zip(dataset_labels, dataset_outputs):
                    print('({:.02f}, {:.02f}) {:.02f}'.format(sample[0], sample[1], np.abs(sample[1] - sample[0])))

        else:
            dataset_accuracy = dataset_correct_class_results / len(dataset)
            printhead('Accuracy = {:.04f}'.format(dataset_accuracy))

            if args.display_outputs:
                for sample in zip(dataset_labels, dataset_outputs):
                    if sample[0] == sample[1]:
                        print(sample, '/')
                    else:
                        print(sample, 'X')

    else:
        if args.model_type == 'regressor':
            print(dataset_abs_errors)
        else:
            print(dataset_outputs)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='image classification')
    parser.add_argument('--dev', default='0',
                        help='cuda device (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='path to output directory in which to save figures.')
    parser.add_argument('--config', default="./config/config_inference.json",
                        help='path to the config file')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='verbose or not')
    parser.add_argument('--graphs', action='store_true',
                        help='save graphs for MAE etc.')
    parser.add_argument('--display_outputs', action='store_true',
                        help='print the labels and predictions to screen.')
    parser.add_argument('--num_dataset_workers', default=4, type=int,
                        help='Number of threads on which to load and process the dataset.')
    parser.add_argument('--model_type', required=True, choices={'classifier', 'regressor'}, type=str,
                        help='Type of model: `regressor` or `classifier`.')
    parser.add_argument('--itn', action='store_true',
                        help='Whether to apply the ITN.')
    parser.add_argument('--stn', action='store_true',
                        help='Whether to apply the STN.')
    args = parser.parse_args()

    # Read the config file
    with open(args.config) as f:
        config = json.load(f)

    # Data and Model
    args.test_set = config['test_set']
    args.task_model = config['task_model']
    assert (type(args.test_set) == str) & (os.path.exists(args.test_set)), "Dataset does not exist at {}".format(args.test_set)
    assert (type(args.task_model) == str) & (os.path.exists(args.task_model)), "Model .pt does not exist at {}".format(args.task_model)

    # Model Parameters
    args.normalizer = torch.tanh if config['normalizer'] == 'tanh' else None
    args.input_shape = config['input_shape']

    args.thresholds = list(np.arange(0.0, 1.0, 0.01))
    args.label_key = config['label_key']

    # ISTN
    args.nf=config['nf']
    args.itn_path = config['itn_path']
    args.stn_path = config['stn_path']
    if args.stn_path:
        args.stn = config['stn']
        args.max_displacement = config['max_displacement']
        args.cp_spacing = config['cp_spacing']

    args.num_classes = None
    if args.model_type == 'classifier':
        args.num_classes = config['num_classes']
        assert (type(args.num_classes)==int), "num_classes should be a single integer"

    # Logging
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    infer(args)
    close_loggers(logging.getLogger(''))