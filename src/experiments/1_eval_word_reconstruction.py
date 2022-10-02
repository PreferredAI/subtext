import argparse
import datetime
import logging
import statistics
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine
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


def main(args):
    train_embs, test_embs = _get_embs(args.full_embs,
                                      num_train=args.train_size,
                                      num_test=args.test_size,
                                      min_len=args.wp_min_len,
                                      max_len=args.wp_max_len)
    train_vocab, test_vocab = train_embs.keys(), test_embs.keys()
    full_embs = {**train_embs, **test_embs}

    piece_embs = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                  [line.strip().split(maxsplit=1) for line in islice(open(args.piece_embs), 1, None)]}
    piece_vocab = piece_embs.keys()

    wp_tree = WordpieceTree.from_vocab(vocab=piece_vocab, unknown_piece=args.unk_tok, decode=args.decode, decode_k=args.decode_k)

    def best_decomp_emb(tok):
        tok_len = len(tok)
        original_emb = full_embs[tok]
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

    logger.warning(f'calculating train reconstruction error with {len(piece_vocab)} wordpieces..')
    recon_targets = [tok for tok in train_vocab if tok not in piece_vocab]  # skip retained embs for efficiency reasons
    train_cosines = [cosine(best_decomp_emb(target), train_embs[target]) for target in tqdm(recon_targets)]

    logger.warning(f'calculating test reconstruction error with {len(piece_vocab)} wordpieces..')
    recon_targets = [tok for tok in test_vocab if tok not in piece_vocab]  # skip retained embs for efficiency reasons
    test_cosines = [cosine(best_decomp_emb(target), test_embs[target]) for target in tqdm(recon_targets)]

    logger.info(f'\nword_reconstruction'
                f'\nfull_embs: {args.full_embs.name}'
                f'\npiece_embs: {args.piece_embs.name}'
                f'\navg cos distance (train): {statistics.mean(train_cosines):0.3f}'
                f'\navg cos distance (test): {statistics.mean(test_cosines):0.3f}')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wordpieces', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked_0.1.wp'))

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

    assert torch.cuda.is_available(), 'CUDA is not available!'
    args.device = torch.device('cuda')
    main(args)
