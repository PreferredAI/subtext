import argparse
import csv
import datetime
import itertools
import json
import logging
import re
import sys
from collections import defaultdict
from itertools import combinations, islice
from pathlib import Path
from random import Random
from timeit import default_timer as timer

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append('..')  # allows (python script) import of utility.py while maintaining IDE navigability
from src.utility import SubtextMinHeap, SubtextNode, WordpieceTree

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):
    out_filename = f'{args.embeddings.stem}#{args.neighbour_sum}#{args.neighbour_alpha:0.3f}' \
                   f'#decomp_{args.decode_k:.0E}' \
                   f'#wplen_{args.wp_min_len}_{args.wp.max_len}' \
                   f'#{args.wp_size}'
    logger.info(f'saving to: {args.out_dir / out_filename}')
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # LOAD EMBEDDINGS (TRAINING DATA)
    _tok_embeddings = [(tok, torch.from_numpy(np.fromstring(emb, dtype=float, sep=' '))) for tok, emb in
                       tqdm([line.strip().split(maxsplit=1) for line in islice(open(args.embeddings), 1, None)])
                       if tok != args.eol_tok]

    unused_tok_pattern = re.compile(r'^\[unused\d*\]$')
    vocab, emb_weights = zip(*[(tok, emb) for tok, emb in _tok_embeddings if not unused_tok_pattern.search(tok)])

    embeddings = torch.nn.EmbeddingBag.from_pretrained(torch.stack(emb_weights).to(args.device), mode='mean')
    emb_idx_lookup = {cmpd: torch.LongTensor([idx]).to(args.device) for idx, cmpd in enumerate(vocab)}

    def _get_offsets(all_parents):
        parent_wps, offsets = [], []
        for parent in all_parents:
            offsets.append(len(parent_wps))
            parent_wps.extend(parent)
        return torch.cat([emb_idx_lookup[p] for p in parent_wps]), torch.LongTensor(offsets).to(args.device)

    # PREPARE TRAINING DATA
    substr_containers = defaultdict(set)  # for each key, return wps containing key
    for tok in vocab:
        substrs = set([tok[x:y] for x, y in combinations(range(len(tok) + 1), r=2)]) - {tok}
        for substr in substrs:
            substr_containers[substr].add(tok)
    vocab_occurrences = {tok: len(substr_containers[tok]) + 1 for tok in vocab}
    substr_containers = {k: sorted(v) for k, v in substr_containers.items()}  # SORT SO FIXED ORDER ACROSS EXPERIMENTS

    # PREPARE CORPUS WEIGHING COUNTS
    if len(_tok_embeddings) == len(vocab):  # not BERT
        # use piece frequencies as counts
        counts = {tok: occurrence for tok, occurrence in vocab_occurrences.items()}
    else:  # BERT, estimate counts instead
        # assume vocab is sorted from most to least common
        logger.warning('using zipf\'s law to approximate BERT counts; assuming 3B corpus')
        denominators = itertools.accumulate((1/i for i in range(1, len(vocab)+1)))
        zipf_freq = {}
        zipf_s = 1
        for rank, (tok, denom) in enumerate(zip(vocab, denominators)):
            num = 1 / ((rank + 1) ** zipf_s)
            zipf_freq[tok] = num / denom  # https://en.wikipedia.org/wiki/Zipf%27s_law#Theoretical_review
        freq_factor = 3E9 / sum([i for i in zipf_freq.values()])
        counts = {tok: int(freq * freq_factor) for tok, freq in zipf_freq.items()}
    del _tok_embeddings  # don't need this anymore

    # filter vocab into reserved and valid
    reserved_pieces, valid_grams = [], []
    for tok in emb_idx_lookup.keys():
        tok_len = len(tok)
        if tok_len == 1 or tok == args.unk_tok:
            reserved_pieces.append(tok)
        elif args.wp_min_len <= tok_len <= args.wp_max_len:
            valid_grams.append(tok)

    logger.info(f'calculating inital costs, this may take a while..')
    costs_heap = SubtextMinHeap([], alpha=args.neighbour_alpha, counts=counts)
    start_timer = timer()
    invalid_grams = []
    superstr_containers = {}  # for each key, return wps used to form in key

    logger.info(f'\tloading wordpiece tree..')
    G: nx.DiGraph = nx.DiGraph()
    wp_tree: WordpieceTree = WordpieceTree.from_vocab(list(vocab), decode='all', decode_k=args.decode_k)
    valid_grams = sorted(valid_grams, key=lambda x: (len(x), x))
    logger.info(f'\tcalculating valid_gram costs..')

    for child in tqdm(valid_grams):
        all_parents = wp_tree.decode(child, skip_self=True)
        superstr_containers[child] = sorted({j for i in all_parents for j in i if len(j) > 1})
        if not all_parents:
            logger.warning(f'could not decode tok: {child}')
            invalid_grams.append(child)
            continue

        parent_indices, offsets = _get_offsets(all_parents)
        tok_emb = embeddings(torch.stack([emb_idx_lookup[child]]))
        candidate_embs = embeddings(parent_indices, offsets)
        all_cossims = F.cosine_similarity(candidate_embs, tok_emb.expand(len(all_parents), -1))
        max_cossim = all_cossims.max().item()
        min_cosdist = 1 - max_cossim
        cost = min_cosdist
        optimal_decode = all_parents[torch.argmax(all_cossims).item()]

        G.add_edges_from([(parent, child) for parents in all_parents for parent in parents])  # add edge from parent to child
        costs_heap.insert(SubtextNode(compound=child,
                                      decode=optimal_decode,
                                      recon_cost=cost,
                                      children_cost=None,
                                      pos_neighbours_cost=None,
                                      neg_neighbours_cost=None))

    end_timer = timer()
    logger.info(f'initial G created in {end_timer - start_timer:0.4f} seconds')

    """
    CALCULATE NEIGHBOUR COST
    """
    logger.info(f'calculating neighbour costs..')
    start_timer = timer()
    seeded_random = Random(42)
    all_candidates = sorted([i for i in set(valid_grams) - set(invalid_grams)])
    position_keys = sorted([i for i in costs_heap.positions.keys()])
    neighbours_pos, neighbours_neg = [], []
    neighbours_pos_cossim, neighbours_neg_cossim = [], []
    neighbours_pos_weights, neighbours_neg_weights = [], []
    pos_reverse_neighbours_lookup, neg_reverse_neighbours_lookup = defaultdict(set), defaultdict(set)  # tok => target that uses tok as a neighbour
    neighbours_bar = tqdm(enumerate(position_keys, start=1), total=len(position_keys))

    supercandidate_counts = []
    util_pos, util_neg = [], []

    for tok_idx, tok in neighbours_bar:
        tok_emb = embeddings(torch.stack([emb_idx_lookup[tok]]))
        supercandidates = superstr_containers[tok] + substr_containers[tok]

        if len(supercandidates) == 0:
            util_pos.append(0), util_neg.append(0)
            rand_candidates = None
            while not rand_candidates or tok in rand_candidates:
                rand_candidates = seeded_random.sample(all_candidates, args.neighbours_candidates)
            rand_embs = embeddings(torch.stack([emb_idx_lookup[c] for c in rand_candidates]))
            rand_cossims = F.cosine_similarity(rand_embs, tok_emb.expand(len(rand_candidates), -1)).tolist()
            candidate_cossim_lookup = {candidate: cossim for candidate, cossim in zip(rand_candidates, rand_cossims)}
            sorted_candidates = sorted(rand_candidates, key=lambda x: -candidate_cossim_lookup[x])
            pos_neighbours = sorted_candidates[:args.neighbours_n]
            neg_neighbours = seeded_random.choices(sorted_candidates[args.neighbours_n:], k=args.neighbours_n)
        elif len(supercandidates) >= args.neighbours_candidates:
            supercandidates = seeded_random.sample(supercandidates, args.neighbours_candidates)
            supercandidate_counts.append(args.neighbours_candidates), util_pos.append(args.neighbours_candidates), util_neg.append(args.neighbours_candidates)
            candidate_embs = embeddings(torch.stack([emb_idx_lookup[c] for c in supercandidates]))
            candidate_cossims = F.cosine_similarity(candidate_embs, tok_emb.expand(len(supercandidates), -1)).tolist()
            candidate_cossim_lookup = {candidate: cossim for candidate, cossim in zip(supercandidates, candidate_cossims)}
            sorted_candidates = sorted(supercandidates, key=lambda x: -candidate_cossim_lookup[x])
            pos_neighbours = sorted_candidates[:args.neighbours_n]
            neg_neighbours = seeded_random.choices(sorted_candidates[args.neighbours_n:], k=args.neighbours_n)
        else:
            supercandidate_counts.append(len(supercandidates))
            supercandidate_embs = embeddings(torch.stack([emb_idx_lookup[c] for c in supercandidates]))
            supercandidate_cossims = F.cosine_similarity(supercandidate_embs, tok_emb.expand(len(supercandidates), -1)).tolist()
            supercandidate_cossim_lookup = {candidate: cossim for candidate, cossim in zip(supercandidates, supercandidate_cossims)}
            sorted_supercandidates = sorted(supercandidates, key=lambda x: -supercandidate_cossim_lookup[x])

            rand_candidates = None
            while not rand_candidates or tok in rand_candidates:
                remaining_candidates = [i for i in all_candidates if i not in supercandidates]
                rand_candidates = seeded_random.sample(remaining_candidates, args.neighbours_candidates)

            rand_embs = embeddings(torch.stack([emb_idx_lookup[c] for c in rand_candidates]))
            rand_cossims = F.cosine_similarity(rand_embs, tok_emb.expand(len(rand_candidates), -1)).tolist()
            rand_cossim_lookup = {candidate: cossim for candidate, cossim in zip(rand_candidates, rand_cossims)}
            sorted_rand_candidates = sorted(rand_candidates, key=lambda x: -rand_cossim_lookup[x])

            pos_neighbours, neg_neighbours = [], []
            temp_vfreq, temp_rand = sorted_supercandidates[0], sorted_rand_candidates[0]
            temp_vfreq_cossim, temp_rand_cossim = supercandidate_cossim_lookup[temp_vfreq], rand_cossim_lookup[temp_rand]

            # merge supercandidates and random_candidates
            while len(pos_neighbours) < args.neighbours_n:
                if temp_vfreq_cossim and temp_vfreq_cossim >= temp_rand_cossim:
                    pos_neighbours.append(temp_vfreq)
                    del sorted_supercandidates[0]
                    temp_vfreq = sorted_supercandidates[0] if sorted_supercandidates else None
                    temp_vfreq_cossim = supercandidate_cossim_lookup[temp_vfreq] if temp_vfreq else None
                else:
                    pos_neighbours.append(temp_rand)
                    del sorted_rand_candidates[0]
                    temp_rand = sorted_rand_candidates[0]
                    temp_rand_cossim = rand_cossim_lookup[temp_rand]

            util_pos.append(len(supercandidates) - len(sorted_supercandidates)), util_neg.append(len(sorted_supercandidates))

            if len(sorted_supercandidates) >= args.neighbours_n:
                neg_neighbours = seeded_random.sample(sorted_supercandidates, args.neighbours_n)
            else:
                neg_neighbours = sorted_supercandidates + sorted_rand_candidates[:args.neighbours_n - len(sorted_supercandidates)]
            candidate_cossim_lookup = {**supercandidate_cossim_lookup, **rand_cossim_lookup}

        pos_neighbours = sorted(pos_neighbours)
        neg_neighbours = sorted(neg_neighbours)
        neighbours_pos.append(pos_neighbours)
        neighbours_neg.append(neg_neighbours)

        pos_cossims = [candidate_cossim_lookup[i] for i in pos_neighbours]
        neg_cossims = [candidate_cossim_lookup[i] for i in neg_neighbours]

        pos_n_cost, neg_n_cost = None, None
        if args.neighbour_sum == 'uniform':
            pos_n_cost, neg_n_cost = sum(pos_cossims), sum(neg_cossims)
            neighbours_pos_weights.append((tok, {tok: 1/args.neighbours_n for tok in pos_neighbours}))
            neighbours_neg_weights.append((tok, {tok: 1/args.neighbours_n for tok in neg_neighbours}))
        elif args.neighbour_sum == 'j_voccur':
            pos_voccur = np.array([vocab_occurrences[i] for i in pos_neighbours])
            weighted_pos_voccur = pos_voccur / sum(pos_voccur)
            pos_neighbour_weights = {tok: weight for tok, weight in zip(pos_neighbours, weighted_pos_voccur)}
            neighbours_pos_weights.append((tok, pos_neighbour_weights))

            neg_voccur = np.array([vocab_occurrences[i] for i in neg_neighbours])
            weighted_neg_voccur = neg_voccur / sum(neg_voccur)
            neg_neighbour_weights = {tok: weight for tok, weight in zip(neg_neighbours, weighted_neg_voccur)}
            neighbours_neg_weights.append((tok, neg_neighbour_weights))

            pos_n_cost, neg_n_cost = sum(pos_cossims * weighted_pos_voccur), sum(neg_cossims * weighted_neg_voccur)

        costs_heap.update(tok, pos_neighbourhood_cost=pos_n_cost, neg_neighbourhood_cost=neg_n_cost)

        neighbours_pos_cossim.append((tok, {n: cossim for n, cossim in zip(pos_neighbours, pos_cossims)}))
        neighbours_neg_cossim.append((tok, {n: cossim for n, cossim in zip(neg_neighbours, neg_cossims)}))

        [pos_reverse_neighbours_lookup[neighbour].add(tok) for neighbour in pos_neighbours]
        [neg_reverse_neighbours_lookup[neighbour].add(tok) for neighbour in neg_neighbours]

    neighbours_df = pd.DataFrame.from_dict({'tok': position_keys, 'pos': neighbours_pos, 'neg': neighbours_neg})
    neighbours_df = neighbours_df.set_index('tok')
    neighbours_df.to_csv(args.out_dir / (out_filename + '.ndf'), quoting=csv.QUOTE_ALL)
    del neighbours_df

    pos_neighbours_df = pd.DataFrame.from_records(neighbours_pos_cossim, columns=['tok', 'neighbour_cossim'])
    neg_neighbours_df = pd.DataFrame.from_records(neighbours_neg_cossim, columns=['tok', 'neighbour_cossim'])

    pos_neighbour_weights_df = pd.DataFrame.from_records(neighbours_pos_weights, columns=['tok', 'neighbour_weights'])
    neg_neighbour_weights_df = pd.DataFrame.from_records(neighbours_neg_weights, columns=['tok', 'neighbour_weights'])

    end_timer = timer()
    logger.info(f'Neighbours created in {end_timer - start_timer:0.4f} seconds')
    """
    CALCULATE NEIGHBOUR COST
    """

    # DEFINE UTIL FUNCTIONS
    def _get_neighbours_cost(tok):
        recon_tok_emb = embeddings(torch.stack([emb_idx_lookup[tok]]))

        pos_neighbour_row = pos_neighbours_df.loc[pos_neighbours_df['tok'] == tok]
        for pos_neighbour in pos_neighbour_row.neighbour_cossim.item().keys():
            pos_neighbour_decodes = wp_tree.decode(pos_neighbour, skip_self=True)
            pos_neighbour_embs = embeddings(*_get_offsets(pos_neighbour_decodes))
            pos_neighbour_cossims = F.cosine_similarity(pos_neighbour_embs, recon_tok_emb.expand(len(pos_neighbour_decodes), -1))
            pos_neighbour_row.neighbour_cossim.item()[pos_neighbour] = max(pos_neighbour_cossims.tolist())

        neg_neighbour_row = neg_neighbours_df.loc[neg_neighbours_df['tok'] == tok]
        for neg_neighbour in neg_neighbour_row.neighbour_cossim.item().keys():
            neg_neighbour_decodes = wp_tree.decode(neg_neighbour, skip_self=True)
            neg_neighbour_embs = embeddings(*_get_offsets(neg_neighbour_decodes))
            neg_neighbour_cossims = F.cosine_similarity(neg_neighbour_embs, recon_tok_emb.expand(len(neg_neighbour_decodes), -1))
            neg_neighbour_row.neighbour_cossim.item()[neg_neighbour] = max(neg_neighbour_cossims.tolist())

        pos_neighbours_cossims = pos_neighbour_row.neighbour_cossim.item()
        neg_neighbours_cossims = neg_neighbour_row.neighbour_cossim.item()

        if args.neighbour_sum == 'uniform':
            return {'pos': sum(pos_neighbours_cossims.values()), 'neg': sum(neg_neighbours_cossims.values())}
        elif args.neighbour_sum == 'j_voccur':
            pos_neighbours = list(pos_neighbours_cossims.keys())
            pos_cossims = [pos_neighbours_cossims[i] for i in pos_neighbours]
            pos_voccur = np.array([vocab_occurrences[i] for i in pos_neighbours])
            weighted_pos_voccur = pos_voccur / sum(pos_voccur)

            neg_neighbours = list(neg_neighbours_cossims.keys())
            neg_cossims = [neg_neighbours_cossims[i] for i in neg_neighbours]
            neg_voccur = np.array([vocab_occurrences[i] for i in neg_neighbours])
            weighted_neg_voccur = neg_voccur / sum(neg_voccur)

            return {'pos': sum(pos_cossims * weighted_pos_voccur), 'neg': sum(neg_cossims * weighted_neg_voccur)}

    def _get_updated_children_cost(update_children, update_parents_lookup):
        child_costs = {}
        for update_child in update_children:
            update_new_parents = update_parents_lookup[update_child]
            update_parent_indices, update_offsets = _get_offsets(update_new_parents)
            update_original_emb = embeddings(torch.stack([emb_idx_lookup[update_child]]))
            update_reconstructed_embs = embeddings(update_parent_indices, update_offsets)
            update_all_cossims = F.cosine_similarity(update_reconstructed_embs,
                                                     update_original_emb.expand(len(update_new_parents), -1))
            update_max_cossim = update_all_cossims.max().item()
            update_recon_cost = 1 - update_max_cossim

            # make sure that we can get a neighbourhood cost later
            if not (pos_neighbours_df['tok'].str.match(f'^{re.escape(update_child)}$').any() and
                    neg_neighbours_df['tok'].str.match(f'^{re.escape(update_child)}$').any()):
                assert False, f'could not get neighbourhood costs for {update_child}'
            update_neighbourhood_cost = _get_neighbours_cost(update_child)

            child_costs[update_child] = {'compound': update_child,
                                         'decode': update_new_parents[torch.argmax(update_all_cossims).item()],
                                         'recon_cost': update_recon_cost,
                                         'pos_neighbourhood_cost': update_neighbourhood_cost['pos'],
                                         'neg_neighbourhood_cost': update_neighbourhood_cost['neg']}
        return child_costs

    def _do_update_neighbours(parent_compound):
        _curr_wps = set([i for i in costs_heap.positions.keys()])
        pos_neighbours_df.drop(pos_neighbours_df.loc[pos_neighbours_df['tok'] == parent_compound].index, inplace=True)
        neg_neighbours_df.drop(neg_neighbours_df.loc[neg_neighbours_df['tok'] == parent_compound].index, inplace=True)
        neighbour_decodes = wp_tree.decode(parent_compound, skip_self=True)
        neighbour_embs = embeddings(*_get_offsets(neighbour_decodes))

        for pos_neighbour in pos_reverse_neighbours_lookup[parent_compound]:
            pos_neighbour_row = pos_neighbours_df.loc[pos_neighbours_df['tok'] == pos_neighbour]
            if pos_neighbour_row.empty:
                continue
            cossim_weights = pos_neighbour_weights_df.loc[pos_neighbour_weights_df['tok'] == pos_neighbour].neighbour_weights.item()
            current_cossims = pos_neighbour_row.neighbour_cossim.item()
            parent_emb = embeddings(torch.stack([emb_idx_lookup[pos_neighbour]]))
            neighbour_cossims = F.cosine_similarity(neighbour_embs, parent_emb.expand(len(neighbour_decodes), -1))
            current_cossims[parent_compound] = max(neighbour_cossims.tolist())
            new_pos_neighbour_cost = sum([cossim * cossim_weights[tok] for tok, cossim in current_cossims.items()])
            costs_heap.update(pos_neighbour, pos_neighbourhood_cost=new_pos_neighbour_cost)

        for neg_neighbour in neg_reverse_neighbours_lookup[parent_compound]:
            neg_neighbour_row = neg_neighbours_df.loc[neg_neighbours_df['tok'] == neg_neighbour]
            if neg_neighbour_row.empty:
                continue
            cossim_weights = neg_neighbour_weights_df.loc[neg_neighbour_weights_df['tok'] == neg_neighbour].neighbour_weights.item()
            current_cossims = neg_neighbour_row.neighbour_cossim.item()
            parent_emb = embeddings(torch.stack([emb_idx_lookup[neg_neighbour]]))
            neighbour_cossims = F.cosine_similarity(neighbour_embs, parent_emb.expand(len(neighbour_decodes), -1))
            current_cossims[parent_compound] = max(neighbour_cossims.tolist())
            new_neg_neighbour_cost = sum([cossim * cossim_weights[tok] for tok, cossim in current_cossims.items()])
            costs_heap.update(neg_neighbour, neg_neighbourhood_cost=new_neg_neighbour_cost)

    records = []
    recon_df = pd.DataFrame({'tok': valid_grams,
                             'decode': valid_grams,
                             'recon_cost': [0] * len(valid_grams)}).set_index('tok')

    def _update_recon_cost():
        for tup in recon_df.itertuples():
            if tup.Index not in wp_tree.wordpiece_vocab_set:
                target_decode = wp_tree.decode(tup.Index, skip_self=True)
                if target_decode != tup.decode:
                    target_parent_indices, target_offsets = _get_offsets(target_decode)
                    target_original_emb = embeddings(torch.stack([emb_idx_lookup[tup.Index]]))
                    target_reconstructed_embs = embeddings(target_parent_indices, target_offsets)
                    target_all_cossims = F.cosine_similarity(target_reconstructed_embs,
                                                             target_original_emb.expand(len(target_decode), -1))
                    target_max_cossim = target_all_cossims.max().item()
                    target_min_cosdist = 1 - target_max_cossim
                    recon_df.loc[tup.Index] = [target_decode, target_min_cosdist]

        return recon_df.recon_cost.sum()

    logger.info('starting greedylazy algorithm..')
    timer_start = timer()
    tqdm_bar = tqdm(total=(costs_heap.size + len(reserved_pieces) - args.wp_size))
    while costs_heap.size and costs_heap.size + len(reserved_pieces) > args.wp_size:
        parent_node = costs_heap.peek()
        parent_compound = parent_node.compound
        children = list(G.successors(parent_compound))
        wp_tree.remove_vocab(parent_compound)

        if not children:
            # update neighbours
            if args.neighbour_alpha > 1e-6:
                _do_update_neighbours(parent_compound)
            records.append((len(records), parent_compound, [], [], {}))
            costs_heap.delete(parent_compound)
            G.remove_edges_from([(grandparent, parent_compound) for grandparent in G.predecessors(parent_compound)])
            tqdm_bar.update(1)
        elif parent_node.children_cost:
            parent_costs = _get_updated_children_cost([parent_compound], {parent_compound: [parent_node.decode]})[parent_compound]
            parent_cost = counts[parent_compound] * (1 - args.neighbour_alpha) * parent_costs['recon_cost']

            for wp in [i for i in parent_node.decode if len(i) > 1]:
                wp_children = [i for i in G.successors(wp) if i != parent_compound]
                if not wp_children:
                    costs_heap.update(wp, children_cost=0)
                else:
                    current_children_cost = costs_heap.get_costs(wp)['children']
                    if current_children_cost is not None:
                        updated_children_cost = current_children_cost - parent_cost
                        costs_heap.update(wp, children_cost=updated_children_cost)

            # update children cost
            new_parents_lookup = {child: wp_tree.decode(child, skip_self=True) for child in children}
            children_costs = _get_updated_children_cost(children, new_parents_lookup)
            for updated_cost in children_costs.values():
                costs_heap.update(updated_cost['compound'],
                                  decode=updated_cost['decode'],
                                  recon_cost=updated_cost['recon_cost'],
                                  pos_neighbourhood_cost=updated_cost['pos_neighbourhood_cost'],
                                  neg_neighbourhood_cost=updated_cost['neg_neighbourhood_cost'])

            # update neighbours
            _do_update_neighbours(parent_compound)

            records.append((len(records), parent_compound, children, new_parents_lookup, children_costs))
            costs_heap.delete(parent_compound)
            G.remove_edges_from([(grandparent, parent_compound) for grandparent in G.predecessors(parent_compound)])
            tqdm_bar.update(1)
        elif children:
            new_parents_lookup = {child: wp_tree.decode(child, skip_self=True) for child in children}
            children_costs = _get_updated_children_cost(children, new_parents_lookup)
            children_cost = sum([counts[child] * (1 - args.neighbour_alpha) * child_cost['recon_cost']
                                 for child, child_cost in children_costs.items()])
            costs_heap.update(parent_compound, children_cost=children_cost)
            wp_tree.add_vocab(parent_compound)  # we want to keep using this compound for future decodes
            tqdm_bar.update(0)
        else:
            assert False, f'CASE invalid: {parent_node}, {children}'
    timer_end = timer()
    tqdm_bar.close()

    model_cost = _update_recon_cost()
    recon_df.to_csv(args.out_dir / (out_filename + '.rdf'), quoting=csv.QUOTE_ALL)

    with open(args.out_dir / (out_filename + '.rec'), 'w') as f:
        for rec in records:
            json.dump(rec, f)
            f.write('\n')

    final_wordpieces = sorted(list(costs_heap.positions.keys()) + reserved_pieces, key=lambda item: (len(item), item))
    with open(args.out_dir / (out_filename + '.wp'), 'w') as f:
        f.write('\n'.join(final_wordpieces))

    with open(args.out_dir / (out_filename + '.wv'), 'w') as f:
        f.write(f'{args.wp_size} 300')
        for tok in final_wordpieces:
            tok_emb = embeddings(torch.stack([emb_idx_lookup[tok]]))[0].tolist()
            f.write(f'\n{tok} {" ".join([f"{i:0.6f}" for i in tok_emb])}')
    logger.warning(f'saved to: {args.out_dir / out_filename}')

    duration = int(timer_end - timer_start)
    timestamp = f'{duration//3600:2.0f}:{(duration % 3600)//60:2.0f}:{duration % 60:2.0f}'
    logger.warning(f'SubText_runtime\t{timestamp}\tmodel_cost\t{model_cost:0.4f}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--embeddings', type=Path, default=Path('../pretrained/word2vec/ascii-masked_trunc_0.1.wv'))
    argparser.add_argument('--counts', type=Path, default=Path('../pretrained/word2vec/ascii-masked.vocab'))

    argparser.add_argument('--out_dir', type=Path, default=Path('DEBUG'))
    argparser.add_argument('--unk_tok', type=str, default='[UNK]')
    argparser.add_argument('--eol_tok', type=str, default='</s>')
    argparser.add_argument('--wp_size', type=int, default=1000)
    # neighbour_cost
    argparser.add_argument('--decode', type=str, choices=['all', 'long_start'], default='all')
    argparser.add_argument('--decode_k', type=int, default=1000)
    argparser.add_argument('--neighbours_candidates', type=int, default='1000')
    argparser.add_argument('--neighbours_n', type=int, default='10')
    argparser.add_argument('--neighbour_sum', choices=['uniform', 'j_voccur'], default='uniform')
    argparser.add_argument('--neighbour_alpha', type=float, default=0.9)

    argparser.add_argument('--wp_min_len', type=int, default=0)
    argparser.add_argument('--wp_max_len', type=int, default=20)
    args = argparser.parse_args()

    now = datetime.datetime.now()
    args.timestamp = '{:02d}{:02d}-{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    print(f'args: {args}')

    assert torch.cuda.is_available(), 'CUDA is not available!'
    args.device = torch.device('cuda')
    main(args)
