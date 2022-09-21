import torch
import pymatreader
import torchio as tio
import numpy as np
from axdata3d.utils import count_samples, Mode

class Dataset3d(torch.utils.data.Dataset):
    def __init__(self, dataset_size=None):
        """
        Args:
        """
        super(Dataset3d, self).__init__()
        self.dataset_size = count_samples() if dataset_size is None else dataset_size
        self.indexes = list(range(self.dataset_size))

class ShuffleDataset3d(Dataset3d):
    def __init__(self, split = 0.8, *args, **kwargs):
        super(ShuffleDataset3d, self).__init__(*args, **kwargs)
        import random
        random.shuffle(self.indexes)
        split_index = int(split * self.dataset_size)
        self.train_indexes = self.indexes[:split_index]
        self.test_indexes = self.indexes[split_index:]

class FeedDataset3d(ShuffleDataset3d):
    def __init__(self, mode=Mode.training, *args, **kwargs):
        super(FeedDataset3d, self).__init__(*args, **kwargs)
        self.mode = mode
        self.preprocess = tio.Resize([64 for _ in range(3)])

    def __len__(self):
        return len(self.train_indexes) if self.mode == Mode.training else len(self.test_indexes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = np.array(idx.tolist())
        # if you are in testing mode shift(+) the index by the number of the training samples
        idx = self.train_indexes[idx] if self.mode == Mode.training else self.test_indexes[idx]
        data = pymatreader.read_mat(f'../../../data/data3d/data3d-{idx}.mat')
        feature = self.preprocess(torch.tensor(np.expand_dims(data['features'], 0)))
        label = self.preprocess(torch.tensor(np.expand_dims(data['labels'], 0)))
        sample = {'data3d': feature,
                  'ideal': label
                  }
        return sample
