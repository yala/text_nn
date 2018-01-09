import json
import pickle
import re

if __name__ == "__main__":
    regex = re.compile('[^a-zA-Z]')
    for mode in ['train', 'dev', 'test']:

        with open('raw_data/snli_1.0/snli_1.0_{}.jsonl'.format(mode), 'r') as f:
            lines = f.readlines()

        raw_data = [ json.loads(line) for line in lines]

        data = []
        for ind, row in enumerate(raw_data):
            if row['gold_label'] == '-':
                continue
            concat_text = '{} \t {}'.format(
                            row['sentence1'].lower(), row['sentence2'].lower())
            data.append({
                        'text1': regex.sub(' ', row['sentence1']),
                        'text2': regex.sub(' ', row['sentence2']),
                        'text' : regex.sub(' ',concat_text),
                        'label': row['gold_label'],
                        'uid': ind
                        })
        pickle.dump(data[:300], open('raw_data/snli_1.0/{}.debug.p'.format(mode),'w'))
        pickle.dump(data, open('raw_data/snli_1.0/{}.p'.format(mode),'w'))
