import gzip
import re
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
from sklearn.datasets import fetch_20newsgroups
import random
random.seed(0)


SMALL_TRAIN_SIZE = 800
CATEGORIES = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

def preprocess_data(data):
    processed_data = []
    for indx, sample in enumerate(data['data']):
        text, label = sample, data['target'][indx]
        label_name = data['target_names'][label]
        text = re.sub('\W+', ' ', text).lower().strip()
        processed_data.append( (text, label, label_name) )
    return processed_data


@RegisterDataset('news_group')
class NewsGroupDataset(AbstractDataset):

    def __init__(self, args, word_to_indx, name, max_length=80):
        self.args = args
        self.args.num_class = 20
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.max_length = max_length
        self.class_balance = {}

        if name in ['train', 'dev']:
            data = preprocess_data(fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=CATEGORIES))
            random.shuffle(data)
            num_train = int(len(data)*.8)
            if name == 'train':
                data = data[:num_train]
            else:
                data = data[num_train:]
        else:
            data = preprocess_data(fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=CATEGORIES))

        for indx, _sample in tqdm.tqdm(enumerate(data)):
            sample = self.processLine(_sample)

            if not sample['y'] in self.class_balance:
                self.class_balance[ sample['y'] ] = 0
            self.class_balance[ sample['y'] ] += 1
            self.dataset.append(sample)

        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("NewsGroup dataset doesn't support balanced sampling")
        if args.objective == 'mse':
            raise NotImplementedError("News Group does not support Regression objective")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, row):
        text, label, label_name = row
        text = " ".join(text.split()[:self.max_length])
        x =  get_indices_tensor(text.split(), self.word_to_indx, self.max_length)
        sample = {'text':text,'x':x, 'y':label, 'y_name': label_name}
        return sample
