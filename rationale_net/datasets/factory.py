NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"

DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


# Depending on arg, build dataset
def get_dataset(args, word_to_indx, truncate_train=False):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    if args.dataset in DATASET_REGISTRY:
        train = DATASET_REGISTRY[args.dataset](args, word_to_indx, 'train')
        dev = DATASET_REGISTRY[args.dataset](args, word_to_indx, 'dev')
        test = DATASET_REGISTRY[args.dataset](args, word_to_indx, 'test')

    return train, dev, test
