from abc import ABCMeta, abstractmethod, abstractproperty
import torch
import torch.utils.data as data
import numpy as np
import json
import pdb
import rationale_net.utils.dataset as utils
import rationale_net.utils.train as train_utils
import sklearn.feature_extraction.text as text
import sklearn.metrics.pairwise as pairwise
import numpy as np
import random
import collections

TRAIN_ONLY_ERR_MSG = "{} only supported for train dataset! Instead saw {}"

class AbstractDataset(data.Dataset):
    __metaclass__ = ABCMeta

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        sample = self.dataset[index]
        return sample
