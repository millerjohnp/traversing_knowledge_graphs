from collections import defaultdict, Counter
import random
import itertools
import warnings
import cPickle as pickle
import sys
from optimize import SparseVector
import util
import platform

import fast_graph_traversal as graph_traversal

__author__ = 'Kelvin Gu'

import numpy as np
from os.path import dirname, abspath, join, isfile


# if running on a local Mac, data_directory is at ../../data
# otherwise, it is the current directory (this is where Codalab will copy dependencies)
if platform.system() == 'Darwin':
    data_directory = join(dirname(dirname(abspath(__file__))), 'data')
else:
    data_directory = ''


class PathQuery(object):
    def __init__(self, s, r, t):
        # at least one of s or t must be a string
        assert isinstance(s, str) or isinstance(t, str)
        assert isinstance(r, tuple)  # tuple rather than list. Tuples are hashable.
        self.s = s  # src word(s)
        self.r = r  # relation word(s)
        self.t = t  # target word(s)

    def __repr__(self):
        rep = '{} {} {}'.format(self.s, self.r, self.t)
        if hasattr(self, 'label'):
            rep += ' {}'.format(self.label)
        return rep

    def __eq__(self, other):
        if not isinstance(other, PathQuery):
            return False
        return self.s == other.s and self.r == other.r and self.t == other.t

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.s.__hash__() + self.r.__hash__() + self.t.__hash__()

    @property
    def r_inverted(self):
        return tuple(invert(rel) for rel in self.r[::-1])

    @staticmethod
    def from_triples(triples):
        return [PathQuery(s, (r,), t) for (s, r, t) in triples]

    @staticmethod
    def flatten(rwqs):
        # flatten batch queries into single queries
        flat = []
        for rwq in rwqs:
            for t, rwp in itertools.izip(rwq.t, rwq.label):
                rwq_new = PathQuery(rwq.s, rwq.r, t)
                rwq_new.label = rwp
                flat.append(rwq_new)
        return flat

    @staticmethod
    def batch(rwqs):
        # group random walk queries having the same source and relation sequence
        grouped = defaultdict(list)
        for rwq in rwqs:
            grouped[(rwq.s, rwq.r)].append(rwq)
        batched = []
        for (s, r), group in grouped.iteritems():
            t = [rwq.t for rwq in group]
            label = [rwq.label for rwq in group]
            rwq_new = PathQuery(s, r, t)
            rwq_new.label = label
            batched.append(rwq_new)
        return batched

    @staticmethod
    def stratify(rwqs):
        # stratified sampling (leaves input untouched)
        num_bins = 100
        samples_per_bin = len(rwqs) / num_bins

        bins = defaultdict(list)
        for rwq in rwqs:
            # NOTE: probability of 1.0 gets its own bin
            bin_idx = int(rwq.label * num_bins)
            bins[bin_idx].append(rwq)

        stratified = []
        for b in bins.values():
            sample = list(np.random.choice(b, samples_per_bin, replace=True))
            stratified.extend(sample)
        return stratified

    @staticmethod
    def stats(pqs):
        ents = Counter()
        rels = Counter()
        paths = Counter()
        lengths = Counter()
        for pq in util.verboserate(pqs):
            ents[pq.s] += 1
            ents[pq.t] += 1
            path = pq.r
            paths[path] += 1
            lengths[len(path)] += 1
            for r in path:
                rels[r] += 1
        return ents, rels, paths, lengths


@util.memoize
def load_glove_vectors(file_path):
    file_path = join(data_directory, file_path)
    wvecs = {}
    print 'loading glove vectors'
    sys.stdout.flush()
    with open(file_path) as f_glove:
        for i, line in enumerate(f_glove):
            elems = line.split()
            word = elems[0]
            vec = np.array([float(x) for x in elems[1:]]).reshape(-1, 1)
            wvecs[word] = vec
            if i % 20000 == 0:
                print i
    print 'done'
    return wvecs


@util.memoize
def load_params(params_path, model_name):
    params_path = join(data_directory, params_path)
    print 'loading params'
    sys.stdout.flush()
    with open(params_path, 'r') as f:
        params = pickle.load(f)
    print 'done'

    if model_name == 'NTN':
        return params

    # add inverted relations to params, if they're not already present
    to_add = SparseVector()
    for (ftype, rel), w in params.iteritems():
        if ftype != 'r':
            continue
        if ('r', invert(rel)) in params:
            continue

        if model_name == 'bilinear':
            w_inv = w.T  # transpose
        elif model_name == 'transE':
            w_inv = -w  # negate
        elif model_name == 'bilinear_diag':
            w_inv = 0.1 * np.random.randn(w.shape[0], w.shape[1])  # random
        else:
            raise ValueError(model_name)
        to_add[('r', invert(rel))] = w_inv
    params += to_add

    return params


def parse_dataset(data_path, dev_mode=False, maximum_examples=float('inf')):
    data_path = join(data_directory, data_path)

    entities = set()
    relations = set()

    def get_examples(name):
        filename = join(data_path, name)
        if not isfile(filename):
            print 'Warning: ', filename, ' not found. Skipping...'
            return None

        examples_arr = list()
        with open(filename, 'r') as f:
            num_examples = 0
            for line in util.verboserate(f):
                if num_examples >= maximum_examples:
                        break
                items = line.split()
                s, path, t = items[:3]
                rels = tuple(path.split(','))
                entities.add(s)
                entities.add(t)
                relations.update(rels)

                if len(items) >= 4:
                    label = items[3]
                else:
                    label = '1'  # if no label, assume positive

                # only add positive examples
                if label == '1':
                    examples_arr.append(PathQuery(s, rels, t))
                    num_examples += 1

        return examples_arr

    def get_triples(queries):
        triples_arr = list()
        for query in queries:
            if len(query.r) == 1:
                triples_arr.append((query.s, str(query.r[0]), query.t))
        return triples_arr

    # add datasets
    print 'loading dataset:', data_path

    attributes = {}

    # use the dev set or the test set
    split = 'dev' if dev_mode else 'test'
    util.metadata('split', split)

    print 'Evaluating on {} set.'.format(split.upper())

    for name in ['train', 'test', 'test_deduction', 'test_induction']:
        attributes[name] = get_examples(name.replace('test', split))

    attributes['entity_list'] = list(entities)
    attributes['relations_list'] = list(relations)

    # add graphs
    triples = {}
    for name in ['train', 'test']:
        triples[name] = get_triples(attributes[name])
    attributes['train_graph'] = Graph(triples['train'])
    attributes['full_graph'] = Graph(triples['train'] + triples['test'])

    return util.Bunch(**attributes)


# defines whether an edge is inverted or not
inverted = lambda r: r[:2] == '**'
invert = lambda r: r[2:] if inverted(r) else '**' + r


class Graph(object):
    def __init__(self, triples):
        self.triples = triples
        neighbors = defaultdict(lambda: defaultdict(set))
        relation_args = defaultdict(lambda: defaultdict(set))

        print 'compiling graph...'
        for s, r, t in triples:
            relation_args[r]['s'].add(s)
            relation_args[r]['t'].add(t)
            neighbors[s][r].add(t)
            neighbors[t][invert(r)].add(s)

        def freeze(d):
            frozen = {}
            for key, subdict in d.iteritems():
                frozen[key] = {}
                for subkey, set_val in subdict.iteritems():
                    frozen[key][subkey] = tuple(set_val)
            return frozen

        # WARNING: both neighbors and relation_args must not have default initialization.
        # Default init is dangerous, because we sometimes perform uniform sampling over
        # all keys in the dictionary. This distribution will get altered if a user asks about
        # entities or relations that weren't present.

        # self.neighbors[start][relation] = (end1, end2, ...)
        # self.relation_args[relation][position] = (ent1, ent2, ...)
        # position is either 's' (domain) or 't' (range)
        self.neighbors = freeze(neighbors)
        self.relation_args = freeze(relation_args)
        self.random_entities = []

        cpp_graph = graph_traversal.Graph()
        for s, r, t in triples:
            cpp_graph.add_edge(s, r, t)
            cpp_graph.add_edge(t, invert(r), s)
        self.cpp_graph = cpp_graph

    def random_walk_probs(self, start, path):
        return self.cpp_graph.exact_random_walk_probs(start, list(path))

    @util.profile
    def walk_all(self, start, path, positive_branch_factor=float('inf')):
        if positive_branch_factor == 0:
            return set()

        approx = positive_branch_factor != float('inf')

        if approx:
            return set(self.cpp_graph.approx_path_traversal(start, list(path), positive_branch_factor))
        else:
            return set(self.cpp_graph.path_traversal(start, list(path)))

    def is_trivial_query(self, start, path):
        return self.cpp_graph.is_trivial_query(start, list(path))

    def type_matching_entities(self, path, position):
        if position == 's':
            r = path[0]
        elif position == 't':
            r = path[-1]
        else:
            raise ValueError(position)

        try:
            if not inverted(r):
                return self.relation_args[r][position]
            else:
                inv_pos = 's' if position == 't' else 't'
                return self.relation_args[invert(r)][inv_pos]
        except KeyError:
            # nothing type-matches
            return tuple()

    def random_walk(self, start, length, no_return=False):
        """
        If no_return, the random walk never revisits the same node. Can sometimes return None, None.
        """
        max_attempts = 1000
        for i in range(max_attempts):

            sampled_path = []
            visited = set()
            current = start
            for k in range(length):
                visited.add(current)

                r = random.choice(self.neighbors[current].keys())
                sampled_path.append(r)

                candidates = self.neighbors[current][r]

                if no_return:
                    current = util.sample_excluding(candidates, visited)
                else:
                    current = random.choice(candidates)

                # no viable next step
                if current is None:
                    break

            # failed to find a viable walk. Try again.
            if current is None:
                continue

            return tuple(sampled_path), current

        return None, None

    @util.profile
    def random_walk_constrained(self, start, path):
        """
        Warning! Can sometimes return None.
        """

        # if start node isn't present we can't take this walk
        if start not in self.neighbors:
            return None

        current = start
        for r in path:
            rels = self.neighbors[current]
            if r not in rels:
                # no viable next steps
                return None
            current = random.choice(rels[r])
        return current

    def random_entity(self):
        if len(self.random_entities) == 0:
            self.random_entities = list(np.random.choice(self.neighbors.keys(), size=20000, replace=True))
        return self.random_entities.pop()

    @util.profile
    def random_path_query(self, length):

        while True:
            # choose initial entity uniformly at random
            source = self.random_entity()

            # sample a random walk
            path, target = self.random_walk(source, length)

            # Failed to find random walk. Try again.
            if path is None:
                continue

            pq = PathQuery(source, path, target)
            return pq

    def relation_stats(self):
        stats = defaultdict(dict)
        rel_counts = Counter(r for s, r, t in self.triples)

        for r, args in self.relation_args.iteritems():
            out_degrees, in_degrees = [], []
            for s in args['s']:
                out_degrees.append(len(self.neighbors[s][r]))
            for t in args['t']:
                in_degrees.append(len(self.neighbors[t][invert(r)]))

            domain = float(len(args['s']))
            range = float(len(args['t']))
            out_degree = np.mean(out_degrees)
            in_degree = np.mean(in_degrees)
            stat = {'avg_out_degree': out_degree,
                    'avg_in_degree': in_degree,
                    'min_degree': min(in_degree, out_degree),
                    'in/out': in_degree / out_degree,
                    'domain': domain,
                    'range': range,
                    'r/d': range / domain,
                    'total': rel_counts[r],
                    'log(total)': np.log(rel_counts[r])
                    }

            # include inverted relation
            inv_stat = {'avg_out_degree': in_degree,
                        'avg_in_degree': out_degree,
                        'min_degree': stat['min_degree'],
                        'in/out': out_degree / in_degree,
                        'domain': range,
                        'range': domain,
                        'r/d': domain / range,
                        'total': stat['total'],
                        'log(total)': stat['log(total)']
                        }

            stats[r] = stat
            stats[invert(r)] = inv_stat

        return stats


class NegativeGenerator(object):
    """Returns a list of sampled negatives """

    def __init__(self, graph, max_negative_samples=10, positive_branch_factor=float('inf'),
                 type_matching_negs=True):

        self.max_negative_samples = max_negative_samples
        self.positive_branch_factor = positive_branch_factor
        self.type_matching_negs = type_matching_negs

        self.graph = graph
        self.candidate_cache = defaultdict(lambda: defaultdict(list))
        self.warning_count = 0

    def _warn(self, msg):
        return # TODO: Warnings disabled for now
        if self.warning_count >= 200:
            return
        warnings.warn(msg)
        self.warning_count += 1

    def _restock_candidates(self, r, position):
        if self.type_matching_negs:
            all_candidates = self.graph.relation_args[r][position]
        else:
            all_candidates = self.graph.neighbors.keys()

        return list(np.random.choice(all_candidates, size=20000, replace=True))

    def _sample_candidates(self, r, position):
        """
        A generator of endless samples for the relation r at the given position
        Performs sampling in batch every once in a while, for speed
        """
        candidates = self.candidate_cache[r][position]
        while True:
            # repopulate samples if you run out
            if len(candidates) == 0:
                try:
                    candidates = self._restock_candidates(r, position)
                    self.candidate_cache[r][position] = candidates
                except KeyError:
                    self._warn('No type-matching candidates for relation {} at position {}.'.format(r, position))
                    return
            yield candidates.pop()

    @staticmethod
    def orient(pq, position):
        """
        Orient which direction we are taking the random walk in
        Return start node, and edges in the proper orientation
        """
        s, r, t = pq.s, pq.r, pq.t
        if position == 't':
            start = s
            path = r
        elif position == 's':
            start = t
            path = pq.r_inverted
        else:
            raise ValueError()
        return start, path

    @util.profile
    def __call__(self, query, position, return_positives=False):
        """
        Returns a unique list of negative nodes
        """
        start, path = self.orient(query, position)

        # get a set of end nodes that are true
        positives = self.graph.walk_all(start, path, self.positive_branch_factor)

        if self.max_negative_samples == float('inf'):
            negatives = self._all_negatives(path, positives)
        else:
            negatives = self._sample_negatives(path, positives)

        if len(negatives) == 0:
            self._warn('No negatives generated for query {} at position {}.'.format(query, position))

        if return_positives:
            return negatives, list(positives)

        return negatives

    def _all_negatives(self, path, positives):
        if self.type_matching_negs:
            candidates = set(self.graph.type_matching_entities(path, 't'))
        else:
            candidates = set(self.graph.neighbors.keys())
        return list(candidates - positives)

    def _sample_negatives(self, path, positives):
        # candidates = an iterable of end nodes that might be false
        last_rel = path[-1]
        if not inverted(last_rel):
            candidates = self._sample_candidates(last_rel, 't')
        else:
            candidates = self._sample_candidates(invert(last_rel), 's')

        # filter out end nodes that are actually true
        negatives = set()
        max_attempts = 10 * self.max_negative_samples
        for attempts, candidate in enumerate(candidates):
            if attempts == max_attempts:
                break
            if len(negatives) == self.max_negative_samples:
                break
            if candidate in positives:
                continue
            negatives.add(candidate)

        return list(negatives)