import argparse
import csv
import datetime
import logging
import random
import string
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
from sklearn import neighbors
from tqdm import tqdm

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDINGS, VOCAB, VOCABSET, FIXED_EMBS, FIXED_VOCAB, WPTREE, DECODELOOKUP, EMBLOOKUP = None, None, None, None, None, None, {}, {}


def load_train_data(data_path):
    rand = random.Random(42)
    tag_toks = defaultdict(list)
    result = {'toks': [], 'labels': []}
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['tag', 'tok', 'pre', 'post'], quoting=csv.QUOTE_ALL)
        [tag_toks[line['tag']].append(line['tok']) for line in reader]
        min_tag_len = min([len(toks) for toks in tag_toks.values()])
        for tag, toks in sorted(tag_toks.items()):
            sampled_toks = rand.sample(toks, min_tag_len)
            result['labels'].extend([tag] * min_tag_len)
            result['toks'].extend(sampled_toks)
    return pd.DataFrame.from_dict(result)


def load_data(data_path, window_size=3):
    result = {'toks': [], 'labels': [], 'context': []}
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['tag', 'choices', 'pre', 'post', 'y'], quoting=csv.QUOTE_ALL)
        for example in reader:
            full_pre = example['pre'].translate(str.maketrans('', '', string.punctuation))    # removes punctuation
            full_post = example['post'].translate(str.maketrans('', '', string.punctuation))  # removes punctuation
            context = full_pre.split()[-window_size:] + full_post.split()[:window_size]
            if context:
                result['toks'].append(example['choices'].split()[int(example['y'])])
                result['labels'].append(example['tag'])
                result['context'].append(context)
    return pd.DataFrame.from_dict(result)


def main(args):
    global EMBEDDINGS, VOCAB, VOCABSET, FIXED_EMBS, FIXED_VOCAB, WPTREE, DECODELOOKUP, EMBLOOKUP
    test_files = [i for i in sorted(args.input_data.parent.iterdir(), key=lambda item: (len(str(item)), str(item)))
                  if args.input_data.name in i.stem and i.suffix == '.all']
    assert test_files, f'no input data files found in {args.input_data.parent} for filename {args.input_data.stem}'

    _EMBEDDINGS = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                   [line.strip().split(maxsplit=1) for line in islice(open(args.piece_embs), 1, None)]}
    EMBEDDINGS = {tok: emb / np.sqrt(sum(emb ** 2)) for tok, emb in _EMBEDDINGS.items()}  # normalized
    VOCAB = {i for i in EMBEDDINGS.keys()}
    VOCABSET = frozenset(EMBEDDINGS.keys())

    _FIXED_EMBS = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                   [line.strip().split(maxsplit=1) for line in islice(open(args.full_embs), 1, None)]}
    FIXED_EMBS = {tok: emb / np.sqrt(sum(emb ** 2)) for tok, emb in _FIXED_EMBS.items()}
    FIXED_VOCAB = frozenset(FIXED_EMBS.keys())
    valid_train_df = load_train_data(args.train_data)
    valid_train_df['train_X'] = valid_train_df.toks.map(lambda x: FIXED_EMBS[x])

    if args.n_neighbours > len(valid_train_df):
        args.n_neighbours = len(valid_train_df)
        logger.warning(f'Expected n_neighbors <= n_samples,  but n_samples = {len(valid_train_df)},'
                       f' n_neighbors = {args.n_neighbours}\nUSING {len(valid_train_df)} NEIGHBOURS INSTEAD!')
    clf = neighbors.KNeighborsClassifier(args.n_neighbours)
    clf.fit(np.stack(valid_train_df.train_X), valid_train_df.labels)

    args.results_dir.mkdir(exist_ok=True, parents=True)
    accuracy, mrrs, total_characters, total_pieces = [], [], 0, 0

    WPTREE = WordpieceTree.from_vocab(vocab=EMBEDDINGS.keys(), unknown_piece=args.unk_tok, decode=args.decode, decode_k=args.decode_k)
    if args.decode == 'all':
        for data_file in sorted(test_files, key=lambda item: (len(str(item)), str(item))):
            this_df = load_data(data_file, window_size=args.window_size)
            for tup in tqdm(this_df.itertuples(), total=len(this_df)):
                if not tup.toks in DECODELOOKUP.keys():
                    context_emb = FIXED_EMBS[tup.toks]
                    min_cosdist, min_cosdist_decomp, min_cosdist_emb = 10, None, None
                    all_decompositions = WPTREE.decode(tup.toks)
                    if min([len(decomp) for decomp in all_decompositions]) == 1:
                        DECODELOOKUP[tup.toks] = [tup.toks]
                        EMBLOOKUP[tup.toks] = context_emb
                        continue
                    if not all_decompositions:
                        all_decompositions = [[args.unk_tok]]
                    tok_len = len(tup.toks)
                    for decomp in all_decompositions:
                        decomp_weights = np.array([len(i) for i in decomp]) / tok_len
                        decomp_emb = np.array([EMBEDDINGS[wp] if wp in VOCAB else EMBEDDINGS[args.unk_tok] for wp in decomp])
                        weighted_emb = np.sum(decomp_weights * decomp_emb.T, axis=1)
                        cos_dist = cosine(context_emb, weighted_emb)
                        if cos_dist < min_cosdist:
                            min_cosdist = cos_dist
                            min_cosdist_decomp = decomp
                            min_cosdist_emb = weighted_emb
                    DECODELOOKUP[tup.toks] = min_cosdist_decomp
                    EMBLOOKUP[tup.toks] = min_cosdist_emb
    elif args.decode == 'long_start':
        for data_file in sorted(test_files, key=lambda item: (len(str(item)), str(item))):
            this_df = load_data(data_file, window_size=args.window_size)
            for tup in tqdm(this_df.itertuples(), total=len(this_df)):
                if not tup.toks in DECODELOOKUP.keys():
                    decomp = WPTREE.decode(tup.toks)
                    decomp_weights = np.array([len(i) for i in decomp]) / len(tup.toks)
                    decomp_emb = np.array([EMBEDDINGS[wp] for wp in decomp])
                    weighted_emb = np.sum(decomp_weights * decomp_emb.T, axis=1)
                    DECODELOOKUP[tup.toks] = decomp
                    EMBLOOKUP[tup.toks] = weighted_emb

    def _to_decoded(df_row):
        return DECODELOOKUP[df_row.toks]
    def _to_emb(df_row):
        return EMBLOOKUP[df_row.toks]

    for test_f in sorted(test_files, key=lambda item: (len(str(item)), str(item))):
        test_df = load_data(test_f, window_size=args.window_size)

        test_df['decoded'] = test_df.apply(_to_decoded, axis=1)
        test_df['test_X'] = test_df.apply(_to_emb, axis=1)

        valid_test_df = test_df.dropna()
        preds = clf.predict(np.stack(valid_test_df.test_X))
        pred_probas = clf.predict_proba(np.stack(valid_test_df.test_X))

        result_filename = f'{args.piece_embs.stem.split("#decomp")[0]}_{test_f.stem}_{args.decode}_{args.decode_k}_{args.n_neighbours}.txt'
        result_f = args.results_dir / result_filename
        logger.info(f'writing results to {result_f.resolve()}')

        correct = 0
        y_reci_ranks = []
        proba_labels = sorted(list({i for i in valid_train_df.labels}))
        with open(result_f, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames=['tok', 'actual', 'pred', 'recon_err', 'decoded', 'decode_maxlen',
                                                       'max_wp_coverage', 'rank', 'proba'])
            csv_writer.writeheader()
            for tok, decoded, test_emb, pred, proba, actual in zip(valid_test_df.toks, valid_test_df.decoded,
                                                                   valid_test_df.test_X, preds, pred_probas,
                                                                   valid_test_df.labels):
                pred_idx = np.argmax(proba).item()
                assert proba_labels[pred_idx] == pred, f'{proba_labels}\n{proba}'
                y_idx = proba_labels.index(actual)
                ranks = rankdata(proba * -1, method='max')  # make high probabilities have low ranks
                y_reci_ranks.append(1 / ranks[y_idx])
                csv_writer.writerow({'tok': tok,
                                     'actual': actual,
                                     'pred': pred,
                                     'recon_err': f'{np.linalg.norm(FIXED_EMBS[tok] - test_emb) if tok in VOCAB else -1:.4f}',
                                     'decoded': '_'.join(decoded),
                                     'decode_maxlen': max([len(i) for i in decoded]),
                                     'max_wp_coverage': f'{max([len(i) for i in decoded]) / len(tok):0.4f}',
                                     'rank': ranks[y_idx],
                                     'proba': '\t'.join([str(p) for p in proba])})
                if pred == actual:
                    correct += 1

        acc = correct / len(test_df)
        mrr = mean(y_reci_ranks)
        mrrs.append(mrr)
        accuracy.append(acc)
        total_characters += test_df.decoded.apply(lambda x: len(''.join(x))).sum()
        total_pieces += test_df.decoded.apply(lambda x: len(x)).sum()

    logger.info(f'\nwindow_{args.window_size},{",".join([str(i) for i in accuracy])},{",".join([str(i) for i in mrrs])},{args.piece_embs.name},'
                f'{args.input_data.parent.stem}_{args.input_data.stem},{args.decode},{args.decode_k},{args.n_neighbours},'
                f'avg_acc:{mean(accuracy):0.6f},avg_mrr:{mean(mrrs):0.6f},'
                f'avg_chars_per_wp:{total_characters / total_pieces:0.6f}')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--decode', type=str, choices=['all', 'long_start'], default='all')
    argparser.add_argument('--decode_k', type=int, default=1000)
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--window_size', type=int, default=7)
    argparser.add_argument('--results_dir', type=Path, default=Path('DEBUG'))

    argparser.add_argument('--train_data', type=Path, default=Path('../data/20news-bydate/enwiki-20211011-masked_0.1/20news-bydate_filtered_win_7_intersect_vocab_classes.all'))
    argparser.add_argument('--input_data', type=Path, default=Path('../data/20news-bydate/enwiki-20211011-masked_0.1/20news-bydate_filtered_win_7_intersect_vocab_db_balanced'))
    argparser.add_argument('--n_neighbours', type=int, default=100)

    argparser.add_argument('--full_embs', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked.wv'))
    argparser.add_argument('--piece_embs', type=Path, default=Path('../data/subtext/enwiki-20211011-masked_0.1_uniform_0.900_voccur#decomp#1000#wpminlen#0#maxlen#20#exp#1#10000.wv'))

    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
