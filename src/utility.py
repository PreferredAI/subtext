import logging
import math
from collections import defaultdict, deque, namedtuple
from itertools import combinations, zip_longest
from pathlib import Path
from typing import Deque, Dict, List, NamedTuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Record(NamedTuple):
    step: int
    parent: str
    children: List[str]
    new_parents: List[List[str]]
    costs: List[float]


HeapNode = namedtuple('HeapNode', ['compound', 'cost'])
SubtextNode = namedtuple('SubtextNode', ['compound',
                                         'decode',
                                         'recon_cost',
                                         'children_cost',
                                         'pos_neighbours_cost',
                                         'neg_neighbours_cost'])


class LabelledMinHeap:
    def __init__(self, elements: List[HeapNode]):
        self.size = 0
        self.heap: Deque[HeapNode] = deque()  # idx of node in heap = node position - 1, range = [0, .., self.size - 1]
        self.positions: Dict[str, int] = {}  # range = [1, .., self.size]
        self.sum_cost = 0
        for e in elements:
            self.insert(e)

    def _float_to_top(self, position) -> None:
        while position != 1:
            parent_position = position // 2
            self.exchange(position, position // 2)  # swap node at idx with its parent
            position = parent_position

    def _float_up(self, position, cost) -> None:
        while position != 1:
            parent_position = position // 2
            parent_cost = self.heap[parent_position - 1].cost  # convert position to heap index
            if parent_cost <= cost:
                break
            self.exchange(position, parent_position)  # swap node at idx with its parent
            position = parent_position

    def insert(self, element: HeapNode) -> None:
        assert element is not None, f'cannot insert empty node!'
        assert element.compound not in self.positions.keys(), f'cannot insert duplicate node {element}'
        self.heap.append(element)
        self.sum_cost += element.cost
        self.size += 1
        position = self.size
        self.positions[element.compound] = position
        self._float_up(position, element.cost)  # restore heap property
        assert self.size == len(self.positions)

    def heapify(self, parent_position=1) -> None:  # if called without arguments, heapify from root down
        if parent_position < 1:  # invalid parent, nothing to heapify
            return
        parent_idx = parent_position - 1

        # check left child
        left_position = 2 * parent_position
        if left_position > self.size:  # position has no child nodes, nothing to heapify
            return
        left_idx = left_position - 1
        parent_node, left_node = self.heap[parent_idx], self.heap[left_idx]
        smallest_position, smallest_cost = (
            parent_position, parent_node.cost) if parent_node.cost <= left_node.cost else (left_position, left_node.cost)

        # check right child
        right_position = left_position + 1
        if right_position <= self.size:
            right_idx = left_position  # right_idx == (right_position - 1) == (left_position + 1 - 1)
            right_node = self.heap[right_idx]
            smallest_position = smallest_position if smallest_cost <= right_node.cost else right_position

        # recursive step
        if smallest_position == parent_position:  # heap property maintained, no action required
            return
        else:
            self.exchange(parent_position, smallest_position)
            self.heapify(smallest_position)  # ensure heap property is maintained for children

    def exchange(self, parent_position, child_position) -> None:
        parent_idx, child_idx = parent_position - 1, child_position - 1
        parent_node, child_node = self.heap[parent_idx], self.heap[child_idx]
        self.positions[parent_node.compound], self.positions[child_node.compound] = child_position, parent_position
        self.heap[parent_idx], self.heap[child_idx] = child_node, parent_node

    def delete(self, compound) -> None:
        compound_position = self.positions[compound]
        compound_idx = compound_position - 1
        self.sum_cost -= self.heap[compound_idx].cost
        self.size -= 1
        self._float_to_top(compound_position)  # move target node to the root

        replacement_node = self.heap.pop()
        del self.positions[compound]
        if self.size > 0:
            self.heap[0] = replacement_node  # replace root node with most expensive node
            self.positions[replacement_node.compound] = 1
            self.heapify(1)  # maintain heap property
        assert self.size == len(
            self.positions), f'{self.size},{len(self.positions)},{len(self.heap)},{self.positions},{self.heap}'

    def pop(self) -> HeapNode:  # special case of delete, where we know the position already
        result = self.heap[0]

        # no need to get position or float to top, as we are removing root
        self.sum_cost -= result.cost
        self.size -= 1

        replacement_node = self.heap.pop()
        if self.size > 0:
            self.heap[0] = replacement_node
            self.positions[replacement_node.compound] = 1
            del self.positions[result.compound]
            self.heapify(1)  # maintain heap property
        assert self.size == len(self.positions)
        return result

    def peek(self, n=0) -> HeapNode:
        assert self.size == len(self.positions)
        return self.heap[n] if n < self.size else None

    def update(self, compound, cost) -> None:
        self.delete(compound)
        self.insert(HeapNode(compound, cost))

    def get_cost(self, compound) -> float:
        assert compound in self.positions.keys(), f'{compound}, {self.positions}, {self.heap}'
        position = self.positions[compound]
        node = self.heap[position - 1]
        return node.cost

    def _heap_sort(self):
        result = []
        while self.size > 0:
            step_node = self.pop()
            result.append(step_node.cost)
        return result


class SubtextMinHeap:
    def __init__(self, elements: List[SubtextNode], alpha=0, counts=None):
        self.alpha = alpha
        self.counts = defaultdict(lambda: 1) if counts is None else counts
        self.size = 0
        self.heap: Deque[SubtextNode] = deque()  # idx of node in heap = node position - 1, range = [0, .., self.size - 1]
        self.positions: Dict[str, int] = {}  # range = [1, .., self.size]
        self.sum_cost = 0
        for e in elements:
            self.insert(e)

    @classmethod
    def load(cls, filename: Union[Path, str]):
        data = np.load(filename, allow_pickle=True)
        obj = cls.__new__(cls)
        obj.alpha = data['alpha'].item()
        obj.counts = {i: j for [i, j] in data['counts']}
        obj.size = data['size'].item()
        obj.heap = deque(SubtextNode(*i) for i in data['heap'])
        obj.positions = {i: j for [i, j] in data['positions']}
        obj.sum_cost = data['sum_cost'].item()
        return obj

    def dump(self, filename: Union[Path, str]):
        data = {'alpha': np.array([self.alpha]),
                'counts': np.array([[i, j] for i, j in self.counts.items()]),
                'size': np.array([self.size]),
                'heap': np.array([[i.compound, i.decode, i.recon_cost,
                                  i.children_cost, i.pos_neighbours_cost, i.neg_neighbours_cost] for i in self.heap]),
                'positions': np.array([[i, j] for i, j in self.positions.items()]),
                'sum_cost': np.array([self.sum_cost])}
        with open(filename, 'wb') as f:
            np.savez_compressed(f, **data)

    def _nodecost(self, node: SubtextNode):
        _node_count = self.counts[node.compound]
        _recon_cost = (1 - self.alpha) * sum(i for i in [node.recon_cost, node.children_cost] if i)
        _neighbourhood_cost = self.alpha * sum(i for i in [node.pos_neighbours_cost] + [node.neg_neighbours_cost] if i)
        return _node_count * (_recon_cost + _neighbourhood_cost)

    def _float_to_top(self, position) -> None:
        while position != 1:
            parent_position = position // 2
            self.exchange(position, position // 2)  # swap node at idx with its parent
            position = parent_position

    def _float_up(self, position, cost) -> None:
        while position != 1:
            parent_position = position // 2
            parent_cost = self._nodecost(self.heap[parent_position - 1])  # convert position to heap index
            if parent_cost <= cost:
                break
            self.exchange(position, parent_position)  # swap node at idx with its parent
            position = parent_position

    def insert(self, element: SubtextNode) -> None:
        assert element is not None, f'cannot insert empty node!'
        assert element.compound not in self.positions.keys(), f'cannot insert duplicate node {element}'
        element_cost = self._nodecost(element)
        self.heap.append(element)
        self.sum_cost += element_cost
        self.size += 1
        position = self.size
        self.positions[element.compound] = position
        self._float_up(position, element_cost)  # restore heap property
        assert self.size == len(self.positions)

    def heapify(self, parent_position=1) -> None:  # if called without arguments, heapify from root down
        if parent_position < 1:  # invalid parent, nothing to heapify
            return
        parent_idx = parent_position - 1

        # check left child
        left_position = 2 * parent_position
        if left_position > self.size:  # position has no child nodes, nothing to heapify
            return
        left_idx = left_position - 1
        parent_cost, left_cost = self._nodecost(self.heap[parent_idx]), self._nodecost(self.heap[left_idx])
        smallest_position, smallest_cost = (parent_position, parent_cost) if parent_cost <= left_cost else (
        left_position, left_cost)

        # check right child
        right_position = left_position + 1
        if right_position <= self.size:
            right_idx = left_position  # right_idx == (right_position - 1) == (left_position + 1 - 1)
            right_cost = self._nodecost(self.heap[right_idx])
            smallest_position = smallest_position if smallest_cost <= right_cost else right_position

        # recursive step
        if smallest_position == parent_position:  # heap property maintained, no action required
            return
        else:
            self.exchange(parent_position, smallest_position)
            self.heapify(smallest_position)  # ensure heap property is maintained for children

    def exchange(self, parent_position, child_position) -> None:
        parent_idx, child_idx = parent_position - 1, child_position - 1
        parent_node, child_node = self.heap[parent_idx], self.heap[child_idx]
        self.positions[parent_node.compound], self.positions[child_node.compound] = child_position, parent_position
        self.heap[parent_idx], self.heap[child_idx] = child_node, parent_node

    def delete(self, compound) -> None:
        compound_position = self.positions[compound]
        compound_idx = compound_position - 1
        self.sum_cost -= self._nodecost(self.heap[compound_idx])
        self.size -= 1
        self._float_to_top(compound_position)  # move target node to the root

        replacement_node = self.heap.pop()
        del self.positions[compound]
        if self.size > 0:
            self.heap[0] = replacement_node  # replace root node with most expensive node
            self.positions[replacement_node.compound] = 1
            self.heapify(1)  # maintain heap property
        assert self.size == len(
            self.positions), f'{self.size},{len(self.positions)},{len(self.heap)},{self.positions},{self.heap}'

    def pop(self) -> SubtextNode:  # special case of delete, where we know the position already
        result = self.heap[0]

        # no need to get position or float to top, as we are removing root
        self.sum_cost -= self._nodecost(result)
        self.size -= 1

        replacement_node = self.heap.pop()
        if self.size > 0:
            self.heap[0] = replacement_node
            self.positions[replacement_node.compound] = 1
            del self.positions[result.compound]
            self.heapify(1)  # maintain heap property
        assert self.size == len(self.positions)
        return result

    def peek(self, n=0) -> SubtextNode:
        assert self.size == len(self.positions)
        return self.heap[n] if n < self.size else None

    def update(self, compound, decode=None, recon_cost=None, children_cost=None, pos_neighbourhood_cost=None, neg_neighbourhood_cost=None) -> None:
        position = self.positions[compound]
        node = self.heap[position - 1]
        update_decode = node.decode if decode is None else decode
        update_recon_cost = node.recon_cost if recon_cost is None else recon_cost
        update_children_cost = node.children_cost if children_cost is None else children_cost
        update_pos_neighbourhood_cost = node.pos_neighbours_cost if pos_neighbourhood_cost is None else pos_neighbourhood_cost
        update_neg_neighbourhood_cost = node.neg_neighbours_cost if neg_neighbourhood_cost is None else neg_neighbourhood_cost
        self.delete(compound)
        self.insert(SubtextNode(compound, update_decode, update_recon_cost, update_children_cost, update_pos_neighbourhood_cost, update_neg_neighbourhood_cost))

    def get_costs(self, compound) -> dict:
        assert compound in self.positions.keys(), f'{compound}, {self.positions}, {self.heap}'
        position = self.positions[compound]
        node = self.heap[position - 1]
        return {'total': self._nodecost(node),
                'recon': node.recon_cost,
                'children': node.children_cost,
                'pos_neighbourhood': node.pos_neighbours_cost,
                'neg_neighbourhood': node.neg_neighbours_cost}


class WordpieceTreeNode:
    def __init__(self, value: str, index: int = None, is_valid_piece: bool = False):
        self.value = value
        self.is_valid_piece = is_valid_piece
        self.index = index
        self.transitions = {}

    def add_transition(self, transition_value, index: int = None, is_valid_piece: bool = False):
        self.transitions[transition_value] = WordpieceTreeNode(transition_value,
                                                               index=index,
                                                               is_valid_piece=is_valid_piece)

    def set_vocab_index(self, index=None):
        assert self.is_valid_piece is False
        self.is_valid_piece = True
        self.index = index

    def unset_vocab_index(self):
        assert self.is_valid_piece is True
        self.is_valid_piece = False
        index = self.index
        self.index = None
        return index


class WordpieceTree:
    def __init__(self, root_node=None, vocab=None, unknown_piece=None, unknown_index=None, decode='all', decode_k=1000):
        if decode == 'longest_first_skip_self':
            self.decode = self._decode_longest_first_skip_self
        elif decode == 'longest_overall_skip_self':
            self.decode = self._decode_longest_overall_skip_self
        elif decode[:28] == 'all_decompositions_skip_self':
            self.decode = self._decode_all_decompositions_skip_self
            self.decode_k = int(decode[29:])
        elif decode[:18] == 'all_decompositions':
            self.decode = self._decode_all_decompositions
            self.decode_k = int(decode[19:])
        elif decode == 'all':
            self.decode = self._all_decode
        elif decode == 'long_start':
            self.decode = self._long_start_decode
        else:
            raise ValueError(f'invalid decode mode: {decode}')
        self.decode = self._all_decode
        self.root_node = root_node if root_node else WordpieceTreeNode('')
        self.unknown_piece = unknown_piece
        self.unknown_index = unknown_index
        self.wordpiece_vocab = list(vocab)
        self.wordpiece_vocab_set = set(self.wordpiece_vocab)

    @classmethod
    def from_vocab(cls, vocab, unknown_piece=None, decode='longest_first', decode_k=1000):
        root_node = WordpieceTreeNode('')
        unknown_index = None
        previous_node = root_node
        # for vocab_index, tok in tqdm(enumerate(vocab), total=len(vocab)):
        for vocab_index, tok in enumerate(vocab):
            if tok == unknown_piece:
                unknown_index = vocab_index
            for suffix_tok in tok[:-1]:  # for each suffix tok in this token
                if suffix_tok not in previous_node.transitions.keys():
                    previous_node.add_transition(suffix_tok)
                previous_node = previous_node.transitions[suffix_tok]
            if tok[-1] in previous_node.transitions:
                previous_node.transitions[tok[-1]].set_vocab_index(vocab_index)
            else:
                previous_node.add_transition(tok[-1], index=vocab_index, is_valid_piece=True)
            previous_node = root_node
        return cls(root_node, vocab=vocab, unknown_piece=unknown_piece, unknown_index=unknown_index, decode=decode, decode_k=decode_k)

    def add_vocab(self, new_tok):
        previous_node = self.root_node

        for suffix_tok in new_tok[:-1]:  # for each suffix tok in this token
            if suffix_tok not in previous_node.transitions.keys():
                previous_node.add_transition(suffix_tok)
            previous_node = previous_node.transitions[suffix_tok]
        if new_tok[-1] in previous_node.transitions:
            previous_node.transitions[new_tok[-1]].set_vocab_index()
        else:
            previous_node.add_transition(new_tok[-1], is_valid_piece=True)

        self.wordpiece_vocab.append(new_tok)
        self.wordpiece_vocab_set.add(new_tok)

        return True

    def remove_vocab(self, tok):
        assert tok in self.wordpiece_vocab, f'cannot remove non-existent tok {tok} from wp_tree vocab'

        previous_node = self.root_node
        for c in tok:
            previous_node = previous_node.transitions[c]
        assert previous_node.is_valid_piece, f'tok {tok} is not a valid wordpiece in wp_tree'
        previous_node.unset_vocab_index()
        self.wordpiece_vocab.remove(tok)
        self.wordpiece_vocab_set.remove(tok)

    def walk_tree(self):
        def _print_transitions(node: WordpieceTreeNode, suffix=''):
            for key, node in node.transitions.items():
                key_suffix = f'{suffix}{key}'
                print(key_suffix)
                _print_transitions(node, suffix=key_suffix)
        _print_transitions(self.root_node)

    def print_graph(self, layout: str = 'planar') -> None:
        layout_lookup = {'spring': nx.layout.spring_layout,
                         'kamada_kawai': nx.layout.kamada_kawai_layout,
                         'planar': nx.layout.planar_layout}
        G = self._to_ngx_multidigraph()
        pos = layout_lookup[layout](G)
        nodes = nx.draw_networkx_nodes(G, pos, node_color='grey')
        edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, width=2)
        labels = nx.draw_networkx_labels(G, pos)
        ax = plt.gca()
        ax.set_axis_off()
        plt.show()

    def _long_start_decode(self, tok: str, skip_self: bool = True) -> list:
        result = []
        while len(tok) > 0:
            active_node = self.root_node
            active_piece, last_valid_piece = '', ''
            for char in tok:
                if skip_self and not result and active_piece == tok[:-1]:
                    break
                elif char in active_node.transitions.keys():
                    active_piece += char
                    active_node = active_node.transitions[char]
                    if active_node.is_valid_piece:
                        last_valid_piece = active_piece
                else:
                    break
            piece_len = len(last_valid_piece)
            if piece_len > 0:
                result.append(last_valid_piece)
                tok = tok[piece_len:]
            else:
                # generally not used, as we retain unigrams in base subtext pieces
                # otherwise, we decode the first character from tok as UNK
                assert self.unknown_piece is not None and self.unknown_index is not None
                result.append(self.unknown_piece)
                tok = tok[1:]
        return result

    def _decode_longest_overall_skip_self(self, tok: str, return_indices=False, unique_indices=False) -> list:

        def _find_longest_piece_skip_self(query):
            if len(query) == 0:
                return [], []
            query_piece, query_piece_len, query_indices, start_idx = None, -1, [], -1

            for substr_idx in range(len(query)):
                active_node, last_valid_node = self.root_node, None
                active_piece, last_valid_piece = '', ''
                for char in query[substr_idx:]:
                    if char in active_node.transitions.keys():
                        active_piece += char
                        active_node = active_node.transitions[char]
                        if active_node.is_valid_piece and active_piece != tok:
                            last_valid_node = active_node
                            last_valid_piece = active_piece
                    else:
                        break
                piece_len = len(last_valid_piece)
                if piece_len > 0 and piece_len > query_piece_len:
                    query_piece, query_piece_len, start_idx = last_valid_piece, piece_len, substr_idx
                    if not unique_indices or last_valid_node.index not in query_indices:
                        query_indices.append(last_valid_node.index)

            if query_piece is None:
                return [], []

            end_idx = start_idx + query_piece_len
            front, back = query[:start_idx], query[end_idx:]
            front_pieces, front_indices = _find_longest_piece_skip_self(front)
            back_pieces, back_indices = _find_longest_piece_skip_self(back)
            query_pieces = [i for pieces in [front_pieces, [query_piece], back_pieces] if pieces is not None for i in pieces]
            if unique_indices:
                return query_pieces, sorted(set(query_indices))
            else:
                return query_pieces, None

        result, result_indices = _find_longest_piece_skip_self(tok)
        if result is None or len(result) == 0:
            return [[], []] if return_indices else []
        elif None in result:
            assert False, f'{result}\n{result_indices}'
        return [result, result_indices] if return_indices else result

    def _decode_all_decompositions_skip_self(self, tok: str) -> list:
        result = []
        tok_len = len(tok)

        valid_k = [k for k in range(1, tok_len)
                   if (math.factorial(tok_len) / (math.factorial(k) * math.factorial(tok_len - k))) <= self.decode_k]
        split_idxs = [list(i) for k in valid_k for i in combinations(range(1, tok_len), k)]
        for split_idx in split_idxs:
            split_idx = [0] + split_idx
            split_wps = [tok[i:j] for i, j in zip(split_idx, split_idx[1:]) if j - i > 0] + [tok[split_idx[-1]:]]
            if set(split_wps) <= self.wordpiece_vocab_set:
                result.append(split_wps)
        return result

    def _decode_all_decompositions(self, tok: str) -> list:
        result = []
        tok_len = len(tok)

        if tok_len == 1:  # short circuit
            if tok not in self.wordpiece_vocab_set:
                logger.warning(f'No decompositions for tok: {tok}; using {self.unknown_piece}')
                return [[self.unknown_piece]]
            return [[tok]]

        valid_k = [k for k in range(1, tok_len)
                   if (math.factorial(tok_len) / (math.factorial(k) * math.factorial(tok_len - k))) <= self.decode_k]
        split_idxs = [[0]] + [list(i) for k in valid_k for i in combinations(range(1, tok_len), k)]
        for split_idx in split_idxs:
            split_idx = [0] + split_idx
            split_wps = [tok[i:j] for i, j in zip(split_idx, split_idx[1:]) if j - i > 0] + [tok[split_idx[-1]:]]
            if set([i for i in split_wps if i]) <= self.wordpiece_vocab_set:
                result.append(split_wps)
        if not result:
            logger.warning(f'No decompositions for tok: {tok}; using {self.unknown_piece}')
            return [[self.unknown_piece]]
        return result

    def _all_decode(self, tok: str, skip_self: bool = False) -> list:
        result = []
        tok_len = len(tok)

        if tok_len == 1 and not skip_self:  # short circuit
            if tok not in self.wordpiece_vocab_set:
                logger.warning(f'No decompositions for tok: {tok}; using {self.unknown_piece}')
                return [[self.unknown_piece]]
            return [[tok]]

        split_idxs = [] if skip_self else [[0]]
        valid_k = [k for k in range(1, tok_len)
                   if (math.factorial(tok_len) / (math.factorial(k) * math.factorial(tok_len - k))) <= self.decode_k]
        split_idxs += [list(i) for k in valid_k for i in combinations(range(1, tok_len), k)]
        for split_idx in split_idxs:
            split_idx = [0] + split_idx
            split_wps = [tok[i:j] for i, j in zip(split_idx, split_idx[1:]) if j - i > 0] + [tok[split_idx[-1]:]]
            if {i for i in split_wps if i} <= self.wordpiece_vocab_set:
                result.append(split_wps)
        if not result:
            logger.warning(f'No decompositions for tok: {tok}; using {self.unknown_piece}')
            return [[self.unknown_piece]]
        return result
