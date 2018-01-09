import torch
import torch.autograd as autograd
import torch.nn.functional as F
import gc
import pprint
from collections import Counter
import sklearn.metrics.pairwise as pairwise
import numpy as np

def get_optimizer(models, args):
    '''
        -models: List of models (such as Generator, classif, memory, etc)
        -args: experiment level config

        returns: torch optimizer over models
    '''
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.Adam(params, lr=args.lr,  weight_decay=args.weight_decay)


def get_x_indx(batch, args, eval_model):
    x_indx = autograd.Variable(batch['x'], volatile=eval_model)
    return x_indx



def get_hard_mask(z, return_ind=False):
    '''
        -z: torch Tensor where each element probablity of element
        being selected
        -args: experiment level config

        returns: A torch variable that is binary mask of z >= .5
    '''
    max_z, ind = torch.max(z, dim=-1)
    if return_ind:
        del z
        return ind
    masked = torch.ge(z, max_z.unsqueeze(-1)).float()
    del z
    return masked

def get_gen_path(model_path):
    '''
        -model_path: path of encoder model

        returns: path of generator
    '''
    return '{}.gen'.format(model_path)

def one_hot(label, num_class):
    vec = torch.zeros( (1, num_class) )
    vec[0][label] = 1
    return vec


def gumbel_softmax(input, temperature, cuda):
    noise = torch.rand(input.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = autograd.Variable(noise)
    if cuda:
        noise = noise.cuda()
    x = (input + noise) / temperature
    x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)
    return x.view_as(input)
