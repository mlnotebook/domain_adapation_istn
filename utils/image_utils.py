import os
import torch
import SimpleITK as sitk
import scipy.ndimage
from PIL import Image

def save_tensor_sample(images, save_name, output_dir, nii=True, png=True, num_to_save=3):
    """
    Saves `num_to_save` images from `images` as .nii.gz and .png.
    Args:
        images: 3D Tensor [B, C, H, W, D] - image set to save.
        save_name: str - prefix for the save file.
        output_dir: str - directory in which to save images.
        nii: bool - whether of not to save the 3D image as .nii.gz.
        png: bool - whether of not to save central slice as .png.
        num_to_save: int - the number of samples to save.
    Returns:
        None - saves images
    """
    for i in range(min(images.shape[0], num_to_save)):
        name_stem = os.path.join(output_dir,  '{}_'.format(i) + save_name)
        im = images.detach().cpu()[i][0]
        if nii:
            sitk.WriteImage(sitk.GetImageFromArray(im), name_stem + '.nii.gz')
        if png:
            im = nii3d_to_pil2d(im)
            im.save(name_stem + '.png')
    return

def norm_0_255(image):
    """ Normalizes the image intensities to fall in the range [0,255]
    Args:
        image - [C,H,W,D] - Tensor or numpy.ndarray containing the image data
    Returns:
        image - [C,H,W,D] - Original image with renomalized intensities.
    """
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    return image

def nii3d_to_pil2d(image, scaling=4, interpolation_order=0):
    """ Normalize 3D image intensity, upscale and extract central slice. Convert to Grayscale.
    Also flips the image to maintain nii.gz orientation.
    Args:
        image - 3D Tensor - the 3D image to be processed.
        upscale_factor - int - the factor by which to scale the H, W and D of the 3D image.
        interpolation_order - int - 0=no interpolation, 1=nearest neighbours, 2=linear.
    Returns:
        im - 2D PIL Grayscale Image.
    """
    im = norm_0_255(image)
    im = im.numpy()
    im = scipy.ndimage.zoom(im, (scaling, scaling, scaling), order=interpolation_order, prefilter=False)
    im = Image.fromarray(im.transpose([1, 2, 0])[:, ::-1, ::-1][:, :, im.shape[-1] // 2]).convert('L')
    return im