import argparse
import datetime
import json
import logging
import sys
from itertools import islice
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd
from blingfire import text_to_words
from scipy.stats import rankdata
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args):
    random = Random(args.rand_seed)
    data = json.load(open(args.input_data))
    records = {'type': [], 'x': [], 'mask_idxs': [], 'label': []}
    for k, v in data.items():
        for example in v:
            records['type'].append(k.replace('oos_', ''))
            x = text_to_words(example[0])
            records['x'].append(x)
            tok_probs = [random.random() if len(tok) > 1 else 2 for tok in x.split()]
            records['mask_idxs'].append([idx for idx, prob in enumerate(tok_probs) if prob <= args.mask_prob])
            records['label'].append(example[1])
    label_lookup = {label: idx for idx, label in enumerate(sorted({*records['label']}))}
    logger.info(f'num_labels: {len(label_lookup)}')
    records['y'] = [label_lookup[i] for i in records['label']]
    df = pd.DataFrame.from_dict(records)

    train_emb = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                 [line.strip().split(maxsplit=1) for line in islice(open(args.train_embs), 1, None)]}
    ignore_vocab = set(train_emb.keys())

    train_df, test_df = df[(df.type == 'train') | (df.type == 'val')], df[df.type == 'test']
    logger.info(f'df sizes: train = {len(train_df)}, test = {len(test_df)}')

    embeddings = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                  [line.strip().split(maxsplit=1) for line in islice(open(args.piece_embs), 1, None)]}
    wp_tree: WordpieceTree = WordpieceTree.from_vocab(sorted(embeddings.keys()), decode='long_start', unknown_piece=args.unk_tok)

    def get_sent_emb(row: pd.Series) -> np.ndarray:
        tok_embs = []
        mask_idxs = row.mask_idxs
        for idx, tok in enumerate(row.x.split()):
            if tok in ignore_vocab and idx not in mask_idxs:
                tok_embs.append(train_emb[tok])
                continue
            # alternative reconstruction (i.e., longest first, skip self)
            decomp = wp_tree.decode(tok)
            decomp_weights = np.array([len(i) if i != args.unk_tok else len(tok) for i in decomp]) / len(tok)
            tok_embs.append(sum(weight * embeddings[wp] for weight, wp in zip(decomp_weights, decomp)))
        return np.average(tok_embs, axis=0)

    train_embs = np.stack(train_df.apply(get_sent_emb, axis=1).to_list(), axis=0)
    test_embs = np.stack(test_df.apply(get_sent_emb, axis=1).to_list(), axis=0)

    clf = KNeighborsClassifier(n_neighbors=args.k)
    clf.fit(train_embs, train_df.y.to_numpy())

    y = test_df.y.tolist()
    y_probs = clf.predict_proba(test_embs)
    y_hat = [np.argmax(i) for i in y_probs]

    acc = sum(1 if i == j else 0 for i, j in zip(y, y_hat)) / len(y)

    prob_ranks = [rankdata(prob, method='min') for prob in (y_probs * -1)]
    ranks = np.array([ranks[y_idx] for y_idx, ranks in zip(y, prob_ranks)])
    mrr = np.mean(1 / ranks)

    logger.info(f'intent_classification\n{args.piece_embs.stem}\t{args.mask_prob:0.2f}\t{args.k}\t{acc:0.3f}\t{mrr:0.3f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--mask_prob', type=float, choices=[0.25, 0.50, 0.75], default=0.5)
    argparser.add_argument('--rand_seed', type=int, default=42)
    argparser.add_argument('--k', type=int, default=100)

    argparser.add_argument('--input_data', type=Path, default=Path('../data/raw/clinc150/data_full.json'))
    argparser.add_argument('--train_embs', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked_0.1.wv'))
    argparser.add_argument('--piece_embs', type=Path, default=Path('../data/subtext/enwiki-20211011-masked_0.1_uniform_0.900_voccur#decomp#1000#wpminlen#0#maxlen#20#exp#1#10000.wv'))

    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
