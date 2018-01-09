import torch
import pdb
import argparse

def tensor_to_numpy(tensor):
    return tensor.data[0]

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Rationale-Net Classifier')
    #setup
    parser.add_argument('--train', action='store_true', default=False, help='Whether or not to train model')
    parser.add_argument('--test', action='store_true', default=False, help='Whether or not to run model on test set')
    # device
    parser.add_argument('--cuda', action='store_true', default=False, help='enable the gpu' )
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs to use.')

    parser.add_argument('--debug_mode', action='store_true', default=False, help='debug mode' )

    parser.add_argument('--class_balance', action='store_true', default=False, help='use balanced samlping for train loaded' )
    # learning
    parser.add_argument('--objective', default='cross_entropy', help='choose which loss objective to use')
    parser.add_argument('--aspect', default='overall', help='which aspect to train/eval on')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('--max_batch_size', type=int, default=2000, help='Max batch size to fit through gpu')
    parser.add_argument('--patience', type=int, default=10, help='Num epochs of no dev progress before half learning rate [default: 10]')
    #paths
    #paths
    parser.add_argument('--save_dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('--results_path', type=str, default='', help='where to dump model config and epoch stats')
    parser.add_argument('--summary_path', type=str, default='results/summary.csv', help='where to dump model config and epoch stats')
    parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot to load[default: None]')
    # data loading
    parser.add_argument('--num_workers' , type=int, default=4, help='num workers for data loader')
    # model
    parser.add_argument('--model_form', type=str, default='cnn', help="Form of model, i.e cnn, rnn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=100, help="Dim of hidden layer")
    parser.add_argument('--num_layers', type=int, default=1, help="Num layers of model_form to use")
    parser.add_argument('--dropout', type=float, default=0.1, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='L2 norm penalty [default: 1e-3]')
    parser.add_argument('--filter_num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--filters', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    # data
    parser.add_argument('--dataset', default='pathology', help='choose which dataset to run on')
    parser.add_argument('--embedding', default='pathology', help='choose what embeddings to use')

    # gumbel
    parser.add_argument('--gumbel_temprature', type=float, default=1, help="Start temprature for gumbel softmax. This is annealed via exponential decay")
    parser.add_argument('--gumbel_decay', type=float, default=1e-5, help="Start temprature for gumbel softmax. This is annealed via linear decay")

    # rationale
    parser.add_argument('--get_rationales',  action='store_true', default=False, help="output attributions for dataset")
    parser.add_argument('--selection_lambda', type=float, default=.01, help="y1 in Gen cost L + y1||z|| + y2|zt - zt-1| + y3|{z}|")
    parser.add_argument('--continuity_lambda', type=float, default=.01, help="y2 in Gen cost L + y1||z|| + y2|zt - zt-1|+ y3|{z}|")
    parser.add_argument('--get_attributions', type=bool, default=False, help="output attributions for dataset")

    parser.add_argument('--num_class', type=int, default=2, help="num classes")




    args = parser.parse_args()

    # update args and print
    args.filters = [int(k) for k in args.filters.split(',')]
    if args.objective == 'mse':
        args.num_class = 1

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    return args


