import os
import sys
header = "SentID\tLabel\tSentence"

def extract(src_fname, tgt_fname, binary=False):
    with open(src_fname) as fin:
      with open(tgt_fname, 'w') as fout:
        print(header, file=fout)
        for line in fin:
            sid, label, sent = line.strip().split('\t')
            if binary:
                if 'positive' in label:
                    label = 'positive'
                elif 'negative' in label:
                    label = 'negative'
                elif label == 'neutral':
                    if 'train' in tgt_fname:
                        continue
                else:
                    raise ValueError()
            print('\t'.join([sid, label, sent]), file=fout)


if len(sys.argv)>1 and sys.argv[1] == 'binary':
    extract('raw_data/trn', './train.tsv', binary=True)
    extract('raw_data/dev', './dev.tsv', binary=True)
    extract('raw_data/tst', './test.tsv', binary=True)
else:
    os.makedirs('../SST-5', exist_ok=True)
    extract('raw_data/trn', '../SST-5/train.tsv')
    extract('raw_data/dev', '../SST-5/dev.tsv')
    extract('raw_data/tst', '../SST-5/test.tsv')

