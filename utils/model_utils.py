import torch

def initialize_weights(model):
    """ Initialize the weights of the convolutional and batch normalization layers of a PyTorch model.
    Args:
        model - PyTorch model - the model whose weights are to be initialized.
    Returns:
        None - the weights are initialized in-place.
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0)
    return

def set_requires_grad(model, requires_grad=True):
    """ Toggles the requires_grad property of each model parameter.
    Args:
        model: PyTorch model
        requires_grad: bool - whether to calculate gradients for the parameters
    Returns:
        None: parameters are modified in-place
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

    return

def get_regressor_output(model, images, labels=None, thresholds=None):
    """ Sets the regressor to eval mode and evaluates it on the given input.
    Args:
        model: PyTorch model.
        input: 5D-Tensor - input to be evaluated by the `model`. Shape [B, C, H, W, D].
        labels: 2D-Tensor - used to calculate the error - [B, 1]
        thresholds: list of float - thresholds to apply on regression output to perform binning.
    Returns:
        output: 2D-Tensor - the result of evaluating `model` on `input` - [B, 1]
        average_error: float - the average MAE over the `input`.
        correct_results: list int - the number of outputs that fall within threshold of `labels`
         for each threshold in `thresholds`.
        maes: np.array of floats - the mae for each output given the `labels`.
    """
    if thresholds is not None:
        assert (type(thresholds) == list) & (len(thresholds) > 0) & all([isinstance(t, float) for t in thresholds]), \
            "thresholds should be a list of floats with len > 0"

    model.eval()
    output = model(images)

    # Get Mean Absolute Error
    if labels is not None:
        abs_errors = torch.abs(labels - output)
        mae = abs_errors.mean().item()

        if thresholds is not None:
            # Allow a tolerance of `threshold` on the output for it to be considered correct.
            correct_results = [0] * len(thresholds)
            for threshold_id, threshold in enumerate(thresholds):
                correct_results[threshold_id] += np.sum([1 if (output[idx]  - threshold) <= labels[idx] <= (output[idx] + threshold) else 0
                                                  for idx, x in enumerate(output)])
            return output, mae, correct_results, abs_errors
        else:
            return output, mae, [], abs_errors

    else:
        return output, [], [], abs_errors



def get_classifier_output(model, images, labels):
    """ Evaluates model on the given input.
    Args:
        model: PyTorch model.
        input: 5D-Tensor - input to be evaluated by the `model`. Shape [B, C, H, W, D].
        labels: 2D-Tensor - calculates the error - [B, 1]
    Returns:
        prediction: 2D-Tensor - model output after thresholding - [B, 1]
        correct_results: list int - the number of outputs that are correctly classified.
        output: 2D-Tensor - the result of evaluating `model` on `input` - [B, 1]
    """
    output = model(images)
    try:
        prediction = (0.5 < output).float()
    except:
        prediction = output.max(1, keepdim=True)[1]

    if labels is not None:
        correct_results = prediction.eq(labels.view_as(prediction)).sum().item()
    else:
        correct_results = []

    return output, correct_results, [], []