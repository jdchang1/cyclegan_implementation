import os
import glob
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LearningRateDecay():
    def __init__(self, num_epochs, offset, decay_start):
        """
        Args:
            num_epochs (int): the max number of epochs
            offset (int): hyperparam for delaying/speeding decay
            decay_start (int): start epoch for linear decay to 0
        """
        self.num_epochs = num_epochs
        self.offset = offset
        self.decay_start = decay_start

    def step(self, epoch):
        """
        Linear decay depending on epoch
        Args:
            epoch (int): current epoch
        """
        if decay_start != num_epochs:
            linear_decay = max(0,epoch+self.offset-self.decay_start)/(self.num_epochs-self.decay_start)
        else:
            linear_decay = 0
        return 1.0 - linear_decay

class DiscriminatorBuffer():
    def __init__(self, buffer_size=50):
        self.data = []
        self.buffer_size = buffer_size

    def push_and_pop(self, datapoint):
        result = []
        for element in datapoint.data:
            element = torch.unsqueeze(element, 0)
            if self.buffer_size > len(self.data):
                self.data.append(element)
                result.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    idx = random.randint(0, self.buffer_size-1)
                    result.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    result.append(element)
        return torch.cat(result)

class ImageDataset(Dataset):
    def __init__(self, data_root, tsfms=None, mode='train'):
        self.tsfm = transforms.Compose(tsfms)
        self.files_A = sorted(glob.glob(os.path.join(data_root, '{}A'.format(mode)) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(data_root, '{}B'.format(mode)) + '/*.*'))

    def __getitem__(self, index):
        A = self.tsfm(Image.open(self.files_A[index % len(self.files_A)]))
        B = self.tsfm(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
        return {'A': A, 'B': B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
