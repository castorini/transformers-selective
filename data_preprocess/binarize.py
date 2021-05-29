

def remove_neutral_in_train():
    with open('../MNLI/train.tsv', 'r') as fin:
        with open('./train.tsv', 'w') as fout:
            for i, line in enumerate(fin):
                if i==0:
                    print(line, end='', file=fout)
                elif line.strip().split('\t')[-1] in ['entailment', 'contradiction']:
                    print(line, end='', file=fout)



remove_neutral_in_train()
                    
