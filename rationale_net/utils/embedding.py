import gzip
import numpy as np
import torch
import pickle
import pdb


NO_EMBEDDING_ERR = "Embedding {} not in EMBEDDING_REGISTRY! Available embeddings are {}"

EMBEDDING_REGISTRY = {}


def RegisterEmbedding(name):
    """Registers a dataset."""

    def decorator(f):
        EMBEDDING_REGISTRY[name] = f
        return f
    return decorator


# Depending on arg, return embeddings
def get_embedding_tensor(args):
    if args.embedding not in EMBEDDING_REGISTRY:
        raise Exception(
            NO_EMBEDDING_ERR.format(args.embedding, EMBEDDING_REGISTRY.keys()))

    if args.embedding in EMBEDDING_REGISTRY:
        embeddings, word_to_indx = EMBEDDING_REGISTRY[args.embedding](args)

    args.embedding_dim = embeddings.shape[1]

    return embeddings, word_to_indx


@RegisterEmbedding('beer')
def getBeerEmbedding(args):
    embedding_path='raw_data/beer_review/review+wiki.filtered.200.txt.gz'
    lines = []
    with gzip.open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

@RegisterEmbedding('glove')
def getGloveEmbedding(args):
    embedding_path='data/embeddings/glove.6B/glove.6B.300d.txt'
    lines = []
    with open(embedding_path) as file:
        lines = file.readlines()
        file.close()
    embedding_tensor = []
    word_to_indx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        if not len(emb) == 300:
            continue
        vector = [float(x) for x in emb ]
        if indx == 0:
            embedding_tensor.append( np.zeros( len(vector) ) )
        embedding_tensor.append(vector)
        word_to_indx[word] = indx+1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx


def get_indices_tensor(text_arr, word_to_indx, max_length):
    '''
    -text_arr: array of word tokens
    -word_to_indx: mapping of word -> index
    -max length of return tokens

    returns tensor of same size as text with each words corresponding
    index
    '''
    nil_indx = 0
    text_indx = [ word_to_indx[x] if x in word_to_indx else nil_indx for x in text_arr][:max_length]
    if len(text_indx) < max_length:
        text_indx.extend( [nil_indx for _ in range(max_length - len(text_indx))])

    x =  torch.LongTensor([text_indx])

    return x
