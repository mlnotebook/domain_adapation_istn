# Domain Adapation with Image and Spatial Transformer Networks

R\. Robinson June 2020

A task model (classifier or regressor) may lose performance when it is trained on one domain and evaluated on another. This work performs DA and simultaneously trains a task model on images `S2T` - _i.e._ images transformed from the source domain `s` to the target domain `T`.

DA is performed by training an [Image and Spatial Transformer Network (ISTN)](https://github.com/biomedia-mira/istn) [1] which allow constrained modifications to be applied to the intensity, _i.e._ luminance, contrast, and spatial transformations. Affine and B-Spline-based STNs are implements. The task model is trained **independenly** of the ISTN such that the trained ISTN can then be used to train other task models.

It was originally used to transform between Brain MRI from two different sites (or scanners). It has also been used to move between transform deformed MNIST ([Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST[2]) domains as proof-of-concept. The point of this work is to show the possibility that DA can be performed with explicit and constrained image-level (luminance, contrast etc.) and spatial transformations which retain some explainability over feature-level DA.

---


### Submitted to MICCAI 2020.

If you intend to use this work please cite the pre-print available at::

Robinson, R. et al. (2020). _Image-level Harmonization of Multi-Site Data using Image-and-Spatial Transformer Networks._ MICCAI 2020. https://arxiv.org/XXXXXXXX


---
### References

[1] Lee, M. C. H., Oktay, O., Schuh, A., Schaap, M., & Glocker, B. (2019). _Image-and-Spatial Transformer Networks for Structure-Guided Image Registration._ In Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 11765 LNCS, pp. 337–345). Springer. https://doi.org/10.1007/978-3-030-32245-8_38

[2] Castro, D. C., Tan, J., Kainz, B., Konukoglu, E., & Glocker, B. (2019). _Morpho-MNIST: Quantitative Assessment and Diagnostics for Representation Learning._ Journal of Machine Learning Research, 20(178), 1–29. https://doi.org/http://hdl.handle.net/10044/1/63396

---

# Usage

Scripts for training the ISTN and task models are provided as well as an inference script. To see the benefit of DA with ISTNs:

1. `train_task_model.py` - Train a task model $Task\sb{S}$ on the source domain `S`.
2. `inference.py` - Run task model inference on `S` and target domain `T`. Usually $Task\sb{S}(T) < Task\sb{S}(T)$ due to domain shift.
3. `da_unidirectional_training.py` - Train the ISTN to transform `S` to `T` giving `S2T`. A new task model $Task\sb{A2B}$ is trained either from scratched or finetuned from $Task\sb{S}$. 
4. `da_bidirectional_training.py` - Same as `unidirectional` but uses 2 ISTNs to train like CycleGAN. 
5. `inference.py` - Run new task model inferenece on `T`: usually model performance is recovered such that $Task\sb{A2B}(T) > $Task\sb{S}(T)$.

## Config Files
The config files in the `config` folder contain all of the variables that are passed to each script. Each config is a `.json` file with keys and values.

1. `config_task_training.json` - variables for training the initial task model.
2. `config_train_unidirectional.json` - variables for training the ISTN and new task model with only 1 ISTN.
3. `config_train_bidirectional.json` - variables for training the ISTN and new task model with 2 ISTNs (CycleGAN).
4. `config_inference.json` - variables for performing intererence.

## Training Task Model (`train_task_model.py`)
The script is invoked with the required options
* `--model_type`: {`classifier`, `regressor`} - the model type to be trained. The appropriate architecture is automatically selected.

_Example Usage_
```bash
python ./train_task_model.py --model_type classifier
```

### Config
| key | type | decription |
|-:|:-:|-|
| `train_set` | `str` | Path to to the `.tsv` file containing the training data filenames. |
| `test_set` | `str` | Path to the `.tsv` file containing the validation data filenames. |
| `label_key` | `str` | The key of the Pandas data-frame column to use as labels during task model training. |
| `augmentation` | `bool` | Whether to apply augmentation `1` or not `0` to the training images. |
| `normalizer` | `str` | Function applied to the input images before being passed to the ISTN. Either `tanh` or `None`. Others can be implemented. |
| `input_shape` | `list ints` | The shape of the input images as a list of ints, e.g. `[64, 64, 64]` for a 3D image. |
| `num_classes` | `int` | Currently `1` only. Performs binary classification. Changes the number of nodes in the final layer of the classification task-model. |
| `nf` | `int` | Number of filters in the first layer of the `discriminator`, `itn`, `stn` and `task` models. Currently all the same - differences can be implemented. |
| `learning_rate` | `float` | The learning rate for the `istn` and `discriminator` models |
| `batch_size` | `int` | The batch size for training the `istn`, `discriminator` and `task` models. |
| `val_interval` | `int` | The number of training epochs between each round of validation. |
| `epochs` | `int` | The total number of training epochs. |


## Training ISTN (`da_unidirectional_training.py` or `da_bidirectional_training.py`)
The script is invoked with the required options
* `--model_type`: {`classifier`, `regressor`} - the model type to be trained. The appropriate architecture is automatically selected.

Optional parameters can be passed:
* `--nii`: saves samples (default=3) on each validation step as 3D `.nii.gz`.
* `--png`: saves samples (default=3) on each validation step as 2D `.png` (central slice of 3D image)
* `--B2A`: swaps the two sites (rather than changing the `config` file.)
* `--dev int`: sets the CUDA device to use.
* `--num_dataset_workers int`: the number of threads to spawn for each training dataset. Validation sets are automatically set to 1 worker each.  

_Example Usage_
```bash
python ./da_adversarial_training.py --model_type classifier --nii --png --dev 1
```

### Config
| key | type | decription |
|-:|:-:|-|
| `task_model` | `str` | If you intend to finetune a previously trained task model rather than train a new one from scratch, provide the path to the `.pt` file here |
| `finetune` | `bool` | Whether to finetune `1` or not `0` from a previously trained model (rather than training from scratch). Must provide path to `task_model` `.pt` |
| `label_key` | `str` | The key of the Pandas data-frame column to use as labels during task model training. |
| `augmentation` | `bool` | Whether to apply augmentation `1` or not `0` to the training images. |
| `normalizer` | `str` | Function applied to the input images before being passed to the ISTN. Either `tanh` or `None`. Others can be implemented. |
| `input_shape` | `list ints` | The shape of the input images as a list of ints, e.g. `[64, 64, 64]` for a 3D image. |
| `num_classes` | `int` | Currently `1` only. Performs binary classification. Changes the number of nodes in the final layer of the classification task-model. |
| `nf` | `int` | Number of filters in the first layer of the `discriminator`, `itn`, `stn` and `task` models. Currently all the same - differences can be implemented. |
| `stn` | `str` | The type of STN to employ, either `bspline`, `affine` or `None`.  |
| `max_displacement` | `float` | If `bspline` STN is used, caps the allowed spatial displacement of the control points. Should be in range `[0.0, 1.0]` which is a proportion of the image (_i.e._ `0.1` is 10% of the image dimension.) |
| `cp_spacing` | `list int` | If `bspline` STN is used, the spacing between control points in each dimension in pixels/voxels, _e.g._ `[8, 8, 8]` |
| `early_stopping_epochs` | `int` | If training performance does not improve by `early_stopping_epochs`, training is terminated and last model saved. |
| `learning_rate` | `float` | The learning rate for the `istn` and `discriminator` models |
| `cyc_weight` | `int` | The weighting factor (lambda) to apply which controls the importance of the `identity` loss. |
| `batch_size` | `int` | The batch size for training the `istn`, `discriminator` and `task` models. |
| `val_interval` | `int` | The number of training epochs between each round of validation. |
| `epochs` | `int` | The total number of training epochs. |
| `gan_loss` | `str` | The adversarial loss function to apply to the `istn` output. One of `bce`, `mse`, `l1`. |
| `idt_loss` | `str` | The identity loss function to apply to the `istn` output. One of `bce`, `mse`, `l1`. |
| `dis_loss` | `str` | The loss function to apply to the `discriminator` output. One of `bce`, `mse`, `l1`. |
| `files` | `list str` | List of paths to files which should be copied to the output directory. |

## Inference (`inference.py`)
The script is invoked with the required options
* `--model_type`: {`classifier`, `regressor`} - the model type to be trained. The appropriate architecture is automatically selected.

Optional parameters can be passed:
* `--itn`: if passed, the ITN in `itn_path` will be applied to the images.
* `--stn`: if passed, the STN in `stn_path` will be applied to the images.

```bash
python ./inferece.py --model_type classifier --itn --stn
```

### Config
| key | type | decription |
|-:|:-:|-|
| `test_set` | `str` | Path to the `.tsv` file containing the validation data filenames. |
| `task_model` | `str` | If you intend to finetune a previously trained task model rather than train a new one from scratch, provide the path to the `.pt` file here |
| `label_key` | `str` | The key of the Pandas data-frame column to use as labels during task model training. |
| `itn_path` | `str` | Path to the trained ITN model `.pt`. |
| `stn` | `str` | The type of STN to employ, either `bspline`, `affine` or `None`.  |
| `stn_path` | `str` | Path to the trained STN model `.pt`. |
| `max_displacement` | `float` | If `bspline` STN is used, caps the allowed spatial displacement of the control points. Should be in range `[0.0, 1.0]` which is a proportion of the image (_i.e._ `0.1` is 10% of the image dimension.) |
| `cp_spacing` | `list int` | If `bspline` STN is used, the spacing between control points in each dimension in pixels/voxels, _e.g._ `[8, 8, 8]` |
| `normalizer` | `str` | Function applied to the input images before being passed to the ISTN. Either `tanh` or `None`. Others can be implemented. |
| `input_shape` | `list ints` | The shape of the input images as a list of ints, e.g. `[64, 64, 64]` for a 3D image. |
| `num_classes` | `int` | Currently `1` only. Performs binary classification. Changes the number of nodes in the final layer of the classification task-model. |
| `nf` | `int` | Number of filters in the first layer of the `discriminator`, `itn`, `stn` and `task` models. Currently all the same - differences can be implemented. |
| `thresholds` | `list float` | For a regressor model, a list of thresholds to be applied to the absolute errors to calculate an accuracy metric for the task model. |
