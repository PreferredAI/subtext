import argparse
import datetime
import logging
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_embs(full: Path, num_train=163_189, num_test=1_000, min_len=0, max_len=20):

    full_lines = ((line.split(maxsplit=1)[0], line) for line in islice(open(full, 'r'), 1, None))

    train_emb_path = Path(f'{full.parent}') / f'{full.stem}_train_{num_train}.wv'
    if not train_emb_path.exists():
        train_lines = []
        while len(train_lines) < num_train:
            tok, tok_line = next(full_lines)
            while not (min_len <= len(tok) <= max_len):
                tok, tok_line = next(full_lines)
            train_lines.append(tok_line)
        with open(train_emb_path, 'w') as f:
            f.write(f'{num_train} 300\n')
            f.writelines(train_lines)

    test_emb_path = Path(f'{full.parent}') / f'{full.stem}_test_{num_test}.wv'
    if not test_emb_path.exists():
        test_lines = []
        while len(test_lines) < num_test:
            tok, tok_line = next(full_lines)
            while not (min_len <= len(tok) <= max_len):
                tok, tok_line = next(full_lines)
            test_lines.append(tok_line)
        with open(test_emb_path, 'w') as f:
            f.write(f'{num_test} 300\n')
            f.writelines(test_lines)

    def _load_embs(emb_path: Path):
        return {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                [line.strip().split(maxsplit=1) for line in islice(open(emb_path), 1, None)]}

    return _load_embs(train_emb_path), _load_embs(test_emb_path)


def _get_wordrelate_df(full: Path, num_test=1_000):
    wordrelate_path = Path(f'{full.parent}') / f'{full.stem}_wordrelate.csv'

    if not wordrelate_path.exists():
        import csv

        from sklearn.metrics import pairwise_distances

        test_emb_path = list(Path(f'{full.parent}').glob(f'{full.stem}_test_{num_test}.wv'))
        assert len(test_emb_path) == 1, f'multiple test_emb_paths found: {test_emb_path}'
        test_emb_path = test_emb_path[0]
        toks, embs = [], []
        for tok, emb in tqdm([line.strip().split(maxsplit=1) for line in islice(open(test_emb_path), 1, None)]):
            toks.append(tok)
            embs.append(np.fromstring(emb, dtype=float, sep=' '))
        assert len(toks) == num_test
        embs = np.array(embs)
        dists = pairwise_distances(embs, metric='cosine')
        results = [(toks[i], toks[j], f'{1 - dists[i,j]:0.5f}')
                   for i in tqdm(range(len(dists)), total=len(dists)) for j in range(len(dists)) if i != j]

        with open(wordrelate_path, 'w') as f:
            writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
            writer.writerow(['Word 1', 'Word 2', 'Human (mean)'])
            writer.writerows(results)

    return pandas.read_csv(wordrelate_path)


def main(args):
    _, test_embs = _get_embs(args.full_embs,
                             num_train=args.train_size,
                             num_test=args.test_size,
                             min_len=args.wp_min_len,
                             max_len=args.wp_max_len)
    ws_df = _get_wordrelate_df(args.full_embs, num_test=args.test_size)

    piece_embs = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                  [line.strip().split(maxsplit=1) for line in islice(open(args.piece_embs), 1, None)]}
    piece_vocab = piece_embs.keys()

    wp_tree = WordpieceTree.from_vocab(vocab=piece_vocab, unknown_piece=args.unk_tok, decode=args.decode, decode_k=args.decode_k)

    def best_decomp_emb(tok):
        tok_len = len(tok)
        original_emb = test_embs[tok]
        min_cosdist, min_cosdist_emb = 10, None
        for decomp in wp_tree.decode(tok):
            decomp_weights = np.array([len(i) for i in decomp]) / tok_len
            decomp_emb = np.array([piece_embs[wp] if wp in piece_vocab else piece_embs[args.unk_tok] for wp in decomp])
            weighted_emb = np.sum(decomp_weights * decomp_emb.T, axis=1)
            cos_dist = cosine(original_emb, weighted_emb)
            if cos_dist < min_cosdist:
                min_cosdist = cos_dist
                min_cosdist_emb = weighted_emb
        return min_cosdist_emb

    logger.info('calculating cosine similarities..')
    ws_df['cossim'] = ws_df.apply(lambda x: 1 - cosine(best_decomp_emb(x['Word 1'].lower()),
                                                       best_decomp_emb(x['Word 2'].lower())),
                                  axis=1)
    pearson_corr, pearson_p = pearsonr(ws_df['Human (mean)'], ws_df['cossim'])
    logger.warning(f'\nword_relate'
                   f'\npiece_embs: {args.piece_embs.stem}'
                   f'\npearson_corr: {pearson_corr:.5f}'
                   f'\npearson_p: {pearson_p:.5f}')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--wp_min_len', type=int, default=0)
    argparser.add_argument('--wp_max_len', type=int, default=20)

    argparser.add_argument('--full_embs', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked.wv'))
    argparser.add_argument('--piece_embs', type=Path, default=Path('../data/subtext/enwiki-20211011-masked_0.1_uniform_0.900_voccur#decomp#1000#wpminlen#0#maxlen#20#exp#1#10000.wv'))
    argparser.add_argument('--decode', type=str, choices=['all', 'long_start'], default='all')
    argparser.add_argument('--decode_k', type=int, default=1000)

    argparser.add_argument('--train_size', type=int, default=163_189)
    argparser.add_argument('--test_size', type=int, default=1_000)

    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
