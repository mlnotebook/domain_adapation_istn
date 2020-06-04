import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters, title, epoch, out_dir):
    """ Plots gradients through different layers in the model during training.
    Args:
        named_parameters: PyTorch dictonary of named model parameters.
        title: str - the title to be used for this particular plot (useful to ID different models).
        epoch: int - the epoch number at which this gradient flow is calculated.
        out_dir: str - the output directory.
    Returns:
        None - plots are created and saved
    """
    ave_grads = []
    max_grads = []
    layers = []
    # Collect the names and gradients of the given parameters
    for pname, parameter in named_parameters:
        if (parameter.requires_grad) and ("bias" not in pname):
            layers.append(pname)
            ave_grads.append(parameter.grad.abs().mean())
            max_grads.append(parameter.grad.abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=1.0, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradients: {}, Epoch {}".format(title, epoch))
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gradients_{}_{}.png'.format(title, epoch)), dpi=300)
    plt.close()

def plot_metric(data_dict, title, metric, args):
    colors=['r', 'b', 'g', 'k']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    for idx, key in enumerate(data_dict.keys()):
        ax.plot(data_dict[key][1], data_dict[key][0], c=colors[idx], label=key)
    ax.legend(loc='upper right')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.set_title(title, fontsize=16)
    ax.grid(which='both', axis='both')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, '{}.png'.format(title).replace(' ', '_')), dpi=300)
    plt.close()
    return

def plot_mae(mae, output_dir='./'):
    """Plots a histgram showing the distribution of MAE.
    Args:
        mae: list floats - a list of MAEs.
        output_dir: str - directory to save the figure.
    Returns:
        None
    """
    assert (type(mae)==np.ndarray) & (all([isinstance(m, float) for m in mae])), "`mae` should be a list of floats."
    assert (os.path.exists(output_dir)), "output directory does not exist at {}".format(output_dir)

    plt.hist(mae, label='after', color='blue',
             bins=2 * int((mae.max() * 100).round() - (mae.min() * 100).round()))

    plt.xlabel('MAE')
    plt.ylabel('Count')
    plt.title('Absolute Error Distribution over Validation Data')
    plt.savefig(os.path.join(output_dir, './maehist.png'))
    plt.close()

    return

def plot_cumulative_threshold(acc_per_threhsold, thresholds, output_dir='./'):
    """Plots a bar chart showing the number of correct results within successive thresholds
    Args:
        correct_results_per_threhsold: 1D np.array ints - a 1D array with 1 element per threshold.
            The number of correctly predicted results for each threshold.
        thresholds: 1D np.array floats - a 1D array of thresholds.
        output_dir: str - directory to save the figure.
    Returns:
        None
    """
    assert (type(acc_per_threhsold) == list) &\
           (all([isinstance(result, float) for result in acc_per_threhsold])),\
        "`acc_per_threhsold` should be a list of floats."
    assert (type(thresholds) == list), "`thresholds` should be a list of floats."
    assert (len(thresholds) == len(acc_per_threhsold)),\
        "`correct_results_per_threhsold` and `thresholds` should be the same length."
    assert (os.path.exists(output_dir)), "output directory does not exist at {}".format(output_dir)

    plt.bar(np.arange(len(thresholds)), np.array(acc_per_threhsold)*100, width=1.0, align='edge')
    plt.xlim(np.ceil(np.max(thresholds)))
    plt.gca().axhline(50, linestyle='--', color='k', alpha=0.5)
    plt.gca().axhline(75, linestyle='--', color='k', alpha=0.5)
    plt.gca().axhline(100, linestyle='--', color='k', alpha=0.5)

    interval = np.round(len(thresholds)*0.1,0).astype(int)
    plt.xticks(np.arange(0, len(thresholds), interval), thresholds[::interval])
    plt.xlabel('Threshold (Years)')
    plt.ylabel('Accuracy (%)')
    plt.title('Absolute Error by Threshold over Validation Data')
    plt.savefig(os.path.join(output_dir, './cumulative_threshold.png'))
    plt.close()

    return
