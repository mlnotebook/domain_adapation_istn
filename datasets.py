import torch
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class NiftyDatasetFromTSV(Dataset):
    """Creates a PyTorch Dataset using a .tsv file of .nii.gz paths and labels."""

    def __init__(self, tsv_file, age_limit=None, data_limit=None, normalizer=None, resampler=None, aug=False, seed=42):
        """
        Args:
            tsv_file: (string) - path to tsv file. The file should contain columns:
                0: (str) - list of .nii.gz filepaths
                1: (str) - list of corresponding masks
                2: (int) - age
                3: (int) - sex - 0 = female, 1 = male
                4: (int) - scanner - all identical if the same scanner.
            data_limit: (int) - the maximum number of samples to include.
            age_limit: (list of int, len = 2) - list of 2 ints [min_age, max_age], e.g. [48, 71]
            normalizer: (string) - if not None, one of `tanh` or `sigmoid` to normalize image intensities.
            aug: (bool) - whether to apply augmentations to the data.
        Returns:
            A PyTorch dataset.
        """
        np.random.seed(seed)

        self.data = pd.read_csv(tsv_file)
        if age_limit:
            self.data = self.data[self.data.iloc[:, 3].astype(float) >= age_limit[0]]
            self.data = self.data[self.data.iloc[:, 3].astype(float) <= age_limit[1]]
        self.data = self.data.reindex(np.random.permutation(self.data.index))
        if data_limit:
            self.data = self.data[:data_limit]

        self.data = self.data[:200]
        self.normalizer = None
        if normalizer:
            self.normalizer = torch.tanh if normalizer=='tanh' else self.normalizer
            self.normalizer = torch.sigmoid if normalizer == 'sigmoid' else self.normalizer

        self.aug = aug

        self.samples = []
        for idx in range(len(self.data)):
            img_path = self.data.iloc[idx, 1]
            mask_path = self.data.iloc[idx, 2]
            age = np.array(self.data.iloc[idx, 3]/100, dtype='float64')
            sex = np.array(self.data.iloc[idx, 4], dtype='float64')
            scanner = np.array(self.data.iloc[idx, 5], dtype='int64')

            sample = {'image': img_path, 'mask': mask_path, 'sex': sex, 'age':age, 'scanner': scanner}
            if self.samples == []:
                self.samples = [sample]
            else:
                self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def do_translate(self, image, dx=0, dy=0, dz=0):
        """ Performs random translation of the input image.
        Args:
            image: (np.array floats) [H, W, D] - the input image to augment.
            dx, dy, dz: (int) - the translation to apply in the x, y and z directions in pixels.
                If dx=dy=dz=0, random translation is applied.
        Returns:
            translated_image: (np.array floats [H, W, D] - the translated input image.
        """
        if not (dx & dy & dz):
            dx, dy, dz = [np.random.randint(-3, 3, 1)[0],
                          np.random.randint(-3, 3, 1)[0],
                          np.random.randint(-3, 1, 1)[0]]

        orig_h, orig_w, orig_d = image.shape
        max_shift = np.max(np.abs([dx, dy, dz]))

        canvas = np.pad(image, max_shift)
        canvas_center = np.array(canvas.shape) // 2

        nx = canvas_center[0] - orig_h//2 + dx
        ny = canvas_center[1] - orig_w//2 + dy
        nz = canvas_center[2] - orig_d//2 + dz

        translated = canvas[nx:nx+orig_h, ny:ny+orig_w, nz:nz+orig_d]

        return translated.astype(np.float32), dx, dy, dz

    def __getitem__(self, item):
        sample = self.samples[item]
        image = sitk.ReadImage(os.path.join('/vol/biomedic2/rdr16/pymira/pymira/apps/data_harmonization', sample['image'][2:]), sitk.sitkFloat32)
        mask = sitk.ReadImage(os.path.join('/vol/biomedic2/rdr16/pymira/pymira/apps/data_harmonization',sample['mask'][2:]), sitk.sitkFloat32)

        image_np = sitk.GetArrayFromImage(image)
        mask_np = sitk.GetArrayFromImage(mask)

        if self.aug:
            if np.random.uniform() < 0.5:
                image_np, dx, dy, dz = self.do_translate(image_np)
                mask_np, _, _, _ = self.do_translate(mask_np, dx, dy, dz)
            if np.random.uniform() < 0.5:
                image_np = np.flip(image_np, 2).astype(np.float32)
                mask_np = np.flip(mask_np, 2).astype(np.float32)

        image = torch.from_numpy(image_np).unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        sex = sample['sex']
        age = sample['age']
        scanner = sample['scanner']

        del image_np, mask_np

        return {'image': image, 'mask': mask, 'sex': sex, 'age': age, 'scanner': scanner}

    def get_sample(self, item):
        return self.samples[item]