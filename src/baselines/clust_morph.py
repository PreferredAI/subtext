import argparse
import datetime
import logging
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from random import Random
from timeit import default_timer as timer

import numpy as np
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_ldist(tok, vocab):
    tok_dists = [levenshtein_distance(tok, i) for i in vocab]
    return np.array(tok_dists, dtype=np.uint8)


def main(args):
    rand = Random(42)

    out_filename = f'{args.embs.stem}-clustmorph-{args.clusters}'
    args.output_dir.mkdir(parents=True, exist_ok=True)

    embs = {tok: emb for tok, emb in
            tqdm([line.strip().split(maxsplit=1) for line in islice(open(args.embs), 1, None)])
            if tok != args.eol_tok}

    reserved_vocab = [i for i in embs.keys() if len(i) == 1 or i == args.unk_tok]
    vocab = [i for i in embs.keys() if i not in reserved_vocab]
    assert len(vocab + reserved_vocab) == len(embs)
    num_clusters = args.clusters - len(reserved_vocab)

    kmedoids_start = timer()

    cached_dists_f = args.output_dir / f'{args.embs.stem}-clustmorph.dists'
    cached_vocab_f = args.output_dir / f'{args.embs.stem}-clustmorph.dvoc'

    # build phase
    if False and cached_dists_f.is_file() and cached_vocab_f.is_file():
        cached_vocab = [i.strip() for i in open(cached_vocab_f)]
        assert cached_vocab == vocab
        logger.info(f'loading dists from: {cached_dists_f}')
        dists = np.load(cached_dists_f)
    else:
        logger.info(f'calculating pairwise distances for {len(vocab)} points on wordform levenshtein')
        with Pool(processes=20) as pool:
            dist_results = [pool.apply_async(_get_ldist, (i, vocab)) for i in vocab]
            dists = np.array([i.get() for i in tqdm(dist_results)])
        np.save(cached_dists_f, dists)
        with open(cached_vocab_f) as f:
            f.write('\n'.join(vocab))

    logger.info(f'selecting {num_clusters} initial medoids randomly from {len(vocab)} points')
    medoids = rand.sample(range(len(vocab)), num_clusters)
    unclustered_points = set(range(len(vocab))) - set(medoids)
    logger.info(f'assigning {len(unclustered_points)} unclustered points to {len(medoids)} medoids..')
    sorted_medoids = sorted(medoids)
    clusters = {i: [i] for i in sorted_medoids}
    current_cluster_cost = 0
    for unclustered_point in tqdm(unclustered_points):
        point_dists = dists[unclustered_point, sorted_medoids]
        nearest_medoid_idx = rand.choice(np.where(point_dists == point_dists.min())[0])
        clusters[sorted_medoids[nearest_medoid_idx]].append(unclustered_point)
        current_cluster_cost += point_dists.min()

    max_iters = 5_000
    curr_iter = 0
    logger.info(f'swap phase: {max_iters} rounds')
    while True:
        if curr_iter > max_iters:
            break
        else:
            curr_iter += 1

        swap_cost_delta = None
        tqdm_cluster_bar = tqdm(rand.sample(clusters.items(), len(clusters)))
        for cluster_medoid, cluster_points in tqdm_cluster_bar:
            cluster_costs = dists[cluster_points][:, cluster_points].astype(dtype=np.single)
            point_costs = cluster_costs.sum(axis=0)
            potential_medoid_idxs = np.where(point_costs == point_costs.min())[0]
            potential_medoids = [cluster_points[i] for i in potential_medoid_idxs]

            if cluster_medoid not in potential_medoids:
                swap_cost_delta = point_costs.min() - point_costs[cluster_points.index(cluster_medoid)]
                swap_medoid = rand.choice(potential_medoids)
                logger.info(f'cluster medoid: {cluster_medoid}, swap: {swap_medoid}')

                swap_medoid_points: list = clusters.pop(cluster_medoid)
                reassignment_cost_delta = 0
                new_clusters = {}

                for remaining_medoid, remaining_medoid_points in clusters.items():
                    remaining_dists = dists[remaining_medoid, remaining_medoid_points].astype(dtype=np.float)
                    swap_dists = dists[swap_medoid, remaining_medoid_points].astype(dtype=np.float)
                    updated_remaining_points = []
                    for p, close_dist, swap_dist in zip(remaining_medoid_points, remaining_dists, swap_dists):
                        p_delta = swap_dist - close_dist
                        if p_delta < 0:
                            reassignment_cost_delta += p_delta
                            swap_medoid_points.append(p)
                        elif p_delta > 0:
                            updated_remaining_points.append(p)
                        else:
                            updated_remaining_points.append(p) if p_delta % 2 == 0 else swap_medoid_points.append(p)
                    new_clusters[remaining_medoid] = updated_remaining_points
                new_clusters[swap_medoid] = swap_medoid_points

                current_cluster_cost = current_cluster_cost + swap_cost_delta + reassignment_cost_delta
                tqdm_cluster_bar.set_postfix_str(f'current cost: {current_cluster_cost} (swap: {swap_cost_delta:.2f}, re: {reassignment_cost_delta:.2f})')
                clusters = new_clusters

        if swap_cost_delta is None:
            logger.warning(f'early stop @ {curr_iter}: no change!')
            break

    kmedoids_end = timer()
    duration = int(kmedoids_end - kmedoids_start)
    logger.warning(f'clust_morph in: {duration//3600:02.0f}:{(duration % 3600)//60:02.0f}:{duration % 60:02.0f}')

    medoid_toks = {vocab[i] for i in clusters.keys()}
    out_toks = [tok for tok in embs.keys() if tok in reserved_vocab or tok in medoid_toks]
    with open(args.output_dir / f'{out_filename}.medoids', 'w') as f:
        f.write('\n'.join(out_toks))

    with open(args.output_dir / f'{out_filename}.wv', 'w') as f:
        f.write(f'{len(out_toks)} 300')
        for tok in out_toks:
            f.write(f'\n{tok} {embs[tok]}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--embs', type=Path, default=Path('/data/title_matching/word2vec_unk1/trunc_0.1/ascii-masked_trunc_0.1.wv'))
    argparser.add_argument('--output_dir', type=Path, default=Path('/data/title_matching/word2vec_unk1_kmedoids'))
    argparser.add_argument('--clusters', type=int, default=50000)
    argparser.add_argument('--dist', type=str, default='levenshtein')
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--eol_tok', type=str, default='</s>')
    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')
    main(args)
