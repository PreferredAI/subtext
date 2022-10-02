import argparse
import datetime
import logging
from itertools import islice
from pathlib import Path
from random import Random
from timeit import default_timer as timer

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

logging.basicConfig()
logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)


def main(args):

    out_filename = f'{args.embs.stem}-clustsem_r{args.reassignment:.0E}-{args.clusters}'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _embs = {tok: np.fromstring(emb, dtype=np.float, sep=' ') for tok, emb in
             tqdm([line.strip().split(maxsplit=1) for line in islice(open(args.embs), 1, None)])
             if tok != args.unk_tok and tok != args.eol_tok}

    reserved_vocab = [i for i in _embs.keys() if len(i) == 1 or i == args.unk_tok]
    vocab = [i for i in _embs.keys() if i not in reserved_vocab]
    assert len(vocab + reserved_vocab) == len(_embs)
    num_clusters = args.clusters - len(reserved_vocab)
    assert num_clusters > 0, f'Insufficient clusters ({num_clusters}) with mem_top_k = {args.mem_top_k}'

    norm_embs = np.array([emb for tok, emb in _embs.items() if tok in vocab])
    normalize(norm_embs)
    assert np.isfinite(norm_embs).all()
    logger.info(f'start clustering for {num_clusters} clusters..')
    kmeans_start = timer()
    km = MiniBatchKMeans(n_clusters=num_clusters,
                         batch_size=(len(vocab) // 3) + 1,
                         max_iter=30_000,
                         max_no_improvement=100,
                         tol=1e-5,
                         random_state=42,
                         verbose=1,
                         compute_labels=True,
                         init_size=(100 * num_clusters),
                         n_init=3,
                         init='k-means++',
                         reassignment_ratio=args.reassignment)
    km.fit(norm_embs)

    rand = Random(42)
    medoids = {tok: f'{" ".join([f"{i:.4f}" for i in _embs[tok]])}' for tok in reserved_vocab}
    for centroid in tqdm(km.cluster_centers_):
        centroid_toks, centroid_norm_embs = zip(*[(tok, emb) for tok, emb, label in
                                                  zip(vocab, norm_embs, km.labels_) if tok not in medoids.keys()])
        centroid_dists = np.array([cosine(centroid, norm_embs) for norm_embs in centroid_norm_embs])
        closest_tok_idx = rand.choice(np.where(centroid_dists == centroid_dists.min())[0])
        closest_tok = centroid_toks[closest_tok_idx]
        if closest_tok not in medoids.keys():
            medoids[centroid_toks[closest_tok_idx]] = f'{" ".join([f"{i:.4f}" for i in centroid_norm_embs[closest_tok_idx]])}'
        else:
            assert False, f'duplicate medoid`: {closest_tok}'
    kmeans_end = timer()
    duration = int(kmeans_end - kmeans_start)

    # OUTPUT
    with open(args.output_dir / (out_filename + '.centroids'), 'w') as f:
        f.write('\n'.join([f'{" ".join([str(i) for i in c])}' for c in km.cluster_centers_]))

    labels = {i: idx for idx, i in enumerate(reserved_vocab)}
    labels.update({i: l + len(reserved_vocab) for i, l in zip(vocab, km.labels_)})
    with open(args.output_dir / (out_filename + '.labels'), 'w') as f:
        f.write('\n'.join([f'{tok}\t{str(labels[tok])}' for tok in _embs.keys()]))
    with open(args.output_dir / (out_filename + '.wv'), 'w') as f:

        f.write(f'{len(medoids)} 300\n')
        f.write('\n'.join(f'{tok} {medoids[tok]}' for tok in _embs.keys() if tok in medoids.keys()))

    print(f'clust_sem in {duration//3600:02.0f}:{(duration % 3600)//60:02.0f}:{duration % 60:02.0f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--embs', type=Path, default=Path('/data/title_matching/word2vec_unk1/trunc_0.01/ascii-masked_trunc_0.01.wv'))
    argparser.add_argument('--output_dir', type=Path, default=Path('/data/title_matching/kmeans_out'))
    argparser.add_argument('--clusters', type=int, default=1000)
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--eol_tok', type=str, default='</s>')
    argparser.add_argument('--reassignment', type=float, default=0.1)
    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
