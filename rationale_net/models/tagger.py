import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import rationale_net.models.cnn as cnn
import pdb

'''
    Implements a CNN with arbitary number of layers for tagging (predicts 0/1 for each token in text if token matches label), no max pool over time.
'''
class Tagger(nn.Module):

    def __init__(self, embeddings, args):
        super(Tagger, self).__init__()
        vocab_size, hidden_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False
        self.args = args
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=False)

        self.hidden = nn.Linear((len(args.filters)*args.filter_num), args.num_tags)
        self.dropout = nn.Dropout(args.dropout)

    
    def forward(self, x_indx, mask):
        '''Given input x_indx of dim (batch_size, 1, max_length), return z (batch, length) such that z
        can act as element-wise mask on x'''
        if self.args.model_form == 'cnn':
            ## embedding layer takes in dim (batch_size, max_length), outputs x of dim (batch_size, max_length, hidden_dim)
            x = self.embedding_layer(x_indx.squeeze(1))
        
            if self.args.cuda:
                x = x.cuda()
            ## switch x to dim (batch_size, hidden_dim, max_length)
            x = torch.transpose(x, 1, 2)
            ## activ of dim (batch_size, len(filters)*filter_num, max_length)
            activ = self.cnn(x)
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(args.model_form))

        ## hidden layer takes activ transposed to dim (batch_size, max_length, len(filters)*filter_num) and outputs logit of dim (batch_size, max_length, num_tags)
        logit = self.hidden(torch.transpose(activ, 1, 2))
        return logit, self.hidden
