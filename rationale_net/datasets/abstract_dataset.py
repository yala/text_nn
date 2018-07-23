from abc import ABCMeta, abstractmethod, abstractproperty
import torch.utils.data as data

TRAIN_ONLY_ERR_MSG = "{} only supported for train dataset! Instead saw {}"

class AbstractDataset(data.Dataset):
    __metaclass__ = ABCMeta

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample
