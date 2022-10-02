import argparse
import datetime
import logging
from pathlib import Path

import regex as re
from transformers import BertModel, BertTokenizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args):
    args.outdir.mkdir(exist_ok=True, parents=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    emb_matrix_npy = model.embeddings.word_embeddings.weight.detach().numpy()
    vocab = [tok for tok, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1])]
    filename = f'bert-base-uncased'

    unused_tok_pattern = re.compile(r'^\[unused\d*\]$')
    corpus_freq_vocab_idxs = [idx for idx, tok in enumerate(vocab) if not unused_tok_pattern.search(tok)]

    # dump full bert word (pieces / vectors)
    with open(args.outdir / f'{filename}.wp', 'w') as f:
        f.write('\n'.join([vocab[tok_idx] for tok_idx in corpus_freq_vocab_idxs]))

    with open(args.outdir / f'{filename}.wv', 'w') as f:
        f.write(f'{len(corpus_freq_vocab_idxs)} {str(emb_matrix_npy.shape[-1])}')
        for tok_idx in corpus_freq_vocab_idxs:
            f.write(f'\n{vocab[tok_idx]} ' + ' '.join([f'{v:0.18f}' for v in emb_matrix_npy[tok_idx]]))

    # dump corpus frequency word (pieces / vectors)
    for corpus_freq_size in [25_000, 20_000, 15_000, 10_000]:
        out_fname = f'{filename}-{corpus_freq_size}'

        with open(args.outdir / f'{out_fname}.wv', 'w') as f:
            f.write(f'{corpus_freq_size} {str(emb_matrix_npy.shape[-1])}')
            for tok_idx in corpus_freq_vocab_idxs[:corpus_freq_size]:
                f.write(f'\n{vocab[tok_idx]} ' + ' '.join([f'{v:0.18f}' for v in emb_matrix_npy[tok_idx]]))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--outdir', type=Path, default=Path('../data/bert-base-uncased'))
    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
