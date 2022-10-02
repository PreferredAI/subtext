import argparse
import csv
import datetime
import logging
import string
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.special import softmax
from scipy.stats import rankdata
from tqdm import tqdm

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDINGS, VOCAB, VOCABSET, WPTREE, DECODELOOKUP, EMBLOOKUP = None, None, None, None, {}, {}


def load_sent_data(data_path, window_size=3):
    result = {'tag': [], 'choices': [], 'context': [], 'y': []}
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['tag', 'choices', 'pre', 'post', 'y'], quoting=csv.QUOTE_ALL)
        for example in reader:
            full_pre = example['pre'].translate(str.maketrans('', '', string.punctuation))    # removes punctuation
            full_post = example['post'].translate(str.maketrans('', '', string.punctuation))  # removes punctuation
            context = full_pre.split()[-window_size:] + full_post.split()[:window_size]
            if context:
                result['tag'].append(example['tag'])
                result['choices'].append(example['choices'].split())
                result['y'].append(int(example['y']))
                result['context'].append(context)
    return pd.DataFrame.from_dict(result)


def main(args):
    global EMBEDDINGS, VOCAB, VOCABSET, FIXED_EMBS, FIXED_VOCAB, WPTREE, DECODELOOKUP, EMBLOOKUP
    data_files = [i for i in sorted(args.input_data.parent.iterdir(), key=lambda item: (len(str(item)), str(item)))
                   if args.input_data.name in i.stem and i.suffix == '.all']
    assert data_files, f'no input data files found in {args.input_data.parent} for filename {args.input_data.stem}'

    _EMBEDDINGS = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                   [line.strip().split(maxsplit=1) for line in islice(open(args.piece_embs), 1, None)]}
    EMBEDDINGS = {tok: emb / np.sqrt(sum(emb ** 2)) for tok, emb in _EMBEDDINGS.items()}  # normalized
    VOCAB = {i for i in EMBEDDINGS.keys()}
    VOCABSET = frozenset(EMBEDDINGS.keys())

    _FIXED_EMBS = {tok: np.fromstring(emb, dtype=float, sep=' ') for tok, emb in
                   [line.strip().split(maxsplit=1) for line in islice(open(args.full_embs), 1, None)]}
    FIXED_EMBS = {tok: emb / np.sqrt(sum(emb ** 2)) for tok, emb in _FIXED_EMBS.items()}
    FIXED_VOCAB = frozenset(FIXED_EMBS.keys())

    def _get_context_emb(df_row):
        query = df_row.context
        try:
            query_type = type(query)
            if query_type == str:
                return FIXED_EMBS[query] if query in FIXED_VOCAB else FIXED_EMBS[args.unk_tok]
            elif query_type == list:
                if query:
                    result = [FIXED_EMBS[tok] if tok in FIXED_VOCAB else FIXED_EMBS[args.unk_tok] for tok in query]
                    return np.array(result).mean(axis=0)
                else:
                    return np.nan
            else:
                assert False, f'invalid query {query}'
        except KeyError:
            assert False, f'could not decode {query} with decode: {args.decode}'

    WPTREE = WordpieceTree.from_vocab(vocab=EMBEDDINGS.keys(), unknown_piece=args.unk_tok, decode=args.decode, decode_k=args.decode_k)
    if args.decode == 'all':
        for data_file in sorted(data_files, key=lambda item: (len(str(item)), str(item))):
            this_df = load_sent_data(data_file, window_size=args.window_size)
            for tup in tqdm(this_df.itertuples(), total=len(this_df)):
                for choice in tup.choices:
                    if not choice in DECODELOOKUP.keys():
                        context_emb = FIXED_EMBS[choice]
                        min_cosdist, min_cosdist_decomp, min_cosdist_emb = 10, None, None
                        all_decompositions = WPTREE.decode(choice)
                        tok_len = len(choice)
                        for decomp in all_decompositions:
                            decomp_weights = np.array([len(i) for i in decomp]) / tok_len
                            decomp_emb = np.array([EMBEDDINGS[wp] if wp in VOCAB else EMBEDDINGS[args.unk_tok] for wp in decomp])
                            weighted_emb = np.sum(decomp_weights * decomp_emb.T, axis=1)
                            cos_dist = cosine(context_emb, weighted_emb)
                            if cos_dist < min_cosdist:
                                min_cosdist = cos_dist
                                min_cosdist_decomp = decomp
                                min_cosdist_emb = weighted_emb
                        DECODELOOKUP[choice] = min_cosdist_decomp
                        EMBLOOKUP[choice] = min_cosdist_emb
        def _get_choices_emb(df_row):
            result = [EMBLOOKUP[choice] for choice in df_row.choices]
            return np.array(result)
    elif args.decode == 'long_start':
        for data_file in sorted(data_files, key=lambda item: (len(str(item)), str(item))):
            this_df = load_sent_data(data_file, window_size=args.window_size)
            for tup in tqdm(this_df.itertuples(), total=len(this_df)):
                choices = tup.choices
                context = list(tup.context) if type(tup.context) != list else tup.context
                for tok in choices + context:
                    if not tok in DECODELOOKUP.keys():
                        decomp = WPTREE.decode(tok)
                        decomp_weights = np.array([len(i) for i in decomp]) / len(tok)
                        decomp_emb = np.array([EMBEDDINGS[wp] for wp in decomp])
                        weighted_emb = np.sum(decomp_weights * decomp_emb.T, axis=1)
                        DECODELOOKUP[tok] = decomp
                        EMBLOOKUP[tok] = weighted_emb
        def _get_choices_emb(df_row):
            result = [EMBLOOKUP[choice] for choice in df_row.choices]
            return np.array(result)

    args.results_dir.mkdir(exist_ok=True, parents=True)
    mrrs = []
    accuracy_at = defaultdict(list)

    for sample_idx, data_f in enumerate(sorted(data_files, key=lambda item: (len(str(item)), str(item)))):
        df = load_sent_data(data_f, window_size=args.window_size)

        df['context_embs'] = df.apply(_get_context_emb, axis=1)
        df['choices_embs'] = df.apply(_get_choices_emb, axis=1)
        df = df.dropna()

        if len(df) == 0:
            logger.warning(f'{args.piece_embs.name},{args.input_data.stem},{args.decode},no examples left after decoding!')
            exit(0)

        df['probs'] = df.apply(lambda df_row: softmax(np.dot(df_row.choices_embs, df_row.context_embs)), axis=1)
        df['y_rank'] = df.apply(lambda df_row: rankdata(df_row.probs * -1, method='max')[int(df_row.y)], axis=1)

        for accuracy_level in range(1, len(df.iloc[0].choices) + 1):
            accuracy = sum([1 if i <= accuracy_level else 0 for i in df.y_rank]) / len(df)
            accuracy_at[accuracy_level].append(accuracy)
            logger.info(f'sample {sample_idx} accuracy@{accuracy_level:02d}: {accuracy:0.5f}')

        mrrs.append(df.y_rank.apply(lambda x: 1/x).mean())
        logger.info(f'sample {sample_idx} mrr: {mrrs[-1]:0.5f}\n')

    mean_accuracy_at = {level: mean(accuracy_at[level]) for level in sorted(accuracy_at.keys())}
    logger.info(f'emb,input,decode,avg_mrr,{",".join([f"accuracy@{k}" for k in sorted(mean_accuracy_at.keys())])}\n'
                f'window_{args.window_size},{",".join([str(i) for i in mrrs])},{args.piece_embs.name},'
                f'{args.input_data.parent.stem}_{args.input_data.stem},{args.decode},{args.decode_k},{mean(mrrs):0.5f},'
                f'{",".join([f"{mean_accuracy_at[i]:0.5f}" for i in sorted(mean_accuracy_at.keys())])}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--decode', type=str, choices=['all', 'long_start'], default='all')
    argparser.add_argument('--decode_k', type=int, default=1000)
    # argparser.add_argument('--decode_type', type=str, choices=['all_decompositions_weighted_1000', 'longest_first_1000'], default='all_decompositions_weighted_1000')
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--window_size', type=int, default=7)
    argparser.add_argument('--results_dir', type=Path, default=Path('DEBUG'))

    argparser.add_argument('--input_data', type=Path, default=Path('../data/20news-bydate/enwiki-20211011-masked_0.1/20news-bydate_filtered_win_7_intersect_vocab_db_balanced'))
    argparser.add_argument('--full_embs', type=Path, default=Path('../pretrained/word2vec/enwiki-20211011-masked.wv'))
    argparser.add_argument('--piece_embs', type=Path, default=Path('../data/subtext/enwiki-20211011-masked_0.1_uniform_0.900_voccur#decomp#1000#wpminlen#0#maxlen#20#exp#1#10000.wv'))

    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
