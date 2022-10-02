import argparse
import datetime
import json
import logging
import sys
from itertools import islice
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import HeapNode, LabelledMinHeap, WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):
    vocab, emb_weights = zip(*[(tok, torch.from_numpy(np.fromstring(emb, dtype=float, sep=' '))) for tok, emb in
                               tqdm([line.strip().split(maxsplit=1) for line in islice(open(args.embeddings), 1, None)])
                               if tok != args.eol_tok])
    emb_idx_lookup = {cmpd: torch.LongTensor([idx]).to(args.device) for idx, cmpd in enumerate(vocab)}
    embeddings = torch.nn.EmbeddingBag.from_pretrained(torch.stack(emb_weights).to(args.device), mode='mean')

    base_pieces, valid_grams = [], []
    base_len = 1 if args.records.stem.split('gram#')[0].split('#')[-1] == 'uni' else 2
    wp_min_len = int(args.records.stem.split('wpminlen#')[1].split('#')[0])
    for tok in emb_idx_lookup.keys():
        tok_len = len(tok)
        if tok_len <= base_len or tok == args.unk_tok:
            base_pieces.append(tok)
        elif tok_len >= wp_min_len:
            valid_grams.append(tok)

    wp_tree = WordpieceTree.from_vocab(base_pieces, decode='long_start')
    costs_heap = LabelledMinHeap([HeapNode(compound=tok, cost=(1E6 + i)) for i, tok in enumerate(base_pieces)])
    logger.info(f'creating G, this may take a while..')
    start_timer = timer()
    for child in tqdm(sorted(valid_grams, key=lambda item: (len(item), item))):
        parents = wp_tree.decode(child, skip_self=True)  # pretend that this tok does not exist in vocab
        original_emb = embeddings(torch.stack([emb_idx_lookup[child]]))[0]
        reconstructed_emb = embeddings(torch.stack([emb_idx_lookup[p] for p in parents], dim=0))[0]
        cost = F.mse_loss(original_emb, reconstructed_emb).item()
        costs_heap.insert(HeapNode(compound=child, cost=cost))
        wp_tree.add_vocab(child)
    end_timer = timer()
    logger.info(f'G created in {end_timer - start_timer:0.4f} seconds: {costs_heap.sum_cost / len(costs_heap.positions):,.4f}')
    del wp_tree

    logger.info(f'reading record from {args.records}')
    records = open(args.records).readlines()

    # OUTPUT RESULTS
    args.out_dir.mkdir(exist_ok=True, parents=True)

    for target_size in sorted(args.wp_sizes, reverse=True):
        num_updates = costs_heap.size - target_size
        for _ in tqdm(range(num_updates), total=num_updates):
            _, parent, children, new_parents, costs = json.loads(records.pop(0))
            costs_heap.delete(parent)

        out_filename = f'{args.records.stem.rsplit("#", maxsplit=1)[0]}#{target_size}'
        logger.info(f'saving to: {args.out_dir / out_filename}')

        final_wordpieces = sorted(costs_heap.positions.keys(), key=lambda item: (len(item), item))
        with open(args.out_dir / f'{out_filename}.wp', 'w') as f:
            f.write('\n'.join(final_wordpieces))

        with open(args.out_dir / f'{out_filename}.wv', 'w') as f:
            f.write(f'{target_size} 300')
            for tok in final_wordpieces:
                f.write(f'\n{tok} {" ".join([f"{i:0.6f}" for i in embeddings(torch.stack([emb_idx_lookup[tok]]))[0].tolist()])}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--embeddings', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked_0.1.wv'))
    argparser.add_argument('--counts', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked-vocab.txt'))
    argparser.add_argument('--records', type=Path, default=Path('../data/subtext/enwiki-20211011-masked_0.1_uniform_0.900_voccur#decomp#1000#wpminlen#0#maxlen#20#exp#1#1000.rec'))
    argparser.add_argument('--decode', type=str, choices=['all', 'long_start'], default='all')
    argparser.add_argument('--decode_k', type=int, default=1000)

    argparser.add_argument('--out_dir', type=Path, default=Path('../data/subtext'))
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--eol_tok', type=str, default='</s>')
    argparser.add_argument('--wp_sizes', action='append', type=int, default=[10_000, 15_000, 20_000, 25_000])

    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(args)
