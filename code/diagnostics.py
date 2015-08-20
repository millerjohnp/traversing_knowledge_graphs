from collections import deque, Counter, defaultdict
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import operator
import pandas as pd
from composition import CompositionalModel
from data import invert, inverted, load_params, parse_dataset, data_directory, PathQuery, NegativeGenerator
from os.path import join
from optimize import Observer
import util
import cPickle as pickle

__author__ = 'Kelvin Gu'


def wordnet_render(pq):
    d = {'_hyponym': 'has_instance',
         '_hypernym': 'has_category',
         '_member_meronym': 'has_part',
         '_member_holonym': 'is_part_of'}

    def t(word):
        try:
            if inverted(word):
                return invert(d[invert(word)])
            else:
                return d[word]
        except KeyError:
            return word

    print pq.s, [t(r) for r in pq.r], pq.t


def examine_ranking(ex, samples, scores, ranks):
    print ex
    rows = zip(samples, scores, ranks)

    print rows[0], '(CORRECT)'

    rows.sort(key=operator.itemgetter(2))
    for i, row in enumerate(rows[:10] + rows[-10:]):
        if i == 10:
            print '...'
        print row
    print


def plot_relation_singular_values(params):
    # retrieve all of the relation matrices
    rel_mats = {}
    for key in params:
        param_type, name = key
        if param_type == 'r':
            rel_mats[name] = params[key]

    # plot singular values
    for rel in rel_mats:
        U, s, V = np.linalg.svd(rel_mats[rel])
        plt.plot(s)

    plt.xlabel('Dimension')
    plt.ylabel('Singular Value')
    plt.title('Singular Values of Relation Matrices')
    plt.legend(rel_mats.keys())


def find_inverses(graph):
    inverse = {}
    # heuristic for finding inverse relation
    for r in graph.relation_args.keys():
        ct = Counter()
        # count co-occurring relations
        for k, s in enumerate(graph.relation_args[r]['s']):
            t = graph.neighbors[s][r][0]  # pick an arbitrary target
            ct.update(graph.neighbors[t].keys())  # look at the target's relations
            if k >= 100:
                break
        # inverse = top co-occurring relation that isn't inverse
        for r2, count in ct.most_common():
            if not inverted(r2):
                inverse[r] = r2
                print r, r2, count
                break


def filter_train_inverses(train_graph, dset):

    inverse = {
        '_member_of_domain_topic': '_synset_domain_topic_of',
        '_member_meronym': '_member_holonym',
        '_derivationally_related_form': '_derivationally_related_form',
        '_member_of_domain_region': '_synset_domain_region_of',
        '_hypernym': '_hyponym',
        '_member_holonym': '_member_meronym',
        '_instance_hypernym': '_instance_hyponym',
        '_synset_domain_topic_of': '_member_of_domain_topic',
        '_hyponym': '_hypernym',
        '_instance_hyponym': '_instance_hypernym',
        '_synset_domain_usage_of': '_member_of_domain_usage',
        '_has_part': '_part_of',
        '_part_of': '_has_part',
        '_synset_domain_region_of': '_member_of_domain_region',
    }

    def has_train_inverse(pq):
        s, r, t = pq.s, pq.r[0], pq.t
        if r not in inverse:
            return False

        try:
            return s in train_graph.neighbors[t][inverse[r]]
        except KeyError:
            return False

    grouped = util.group(dset, has_train_inverse)
    easy, hard = grouped[True], grouped[False]

    return easy, hard


def plot_frequency_change(ctr_a, ctr_b):
    keys = set(ctr_a.keys()) | set(ctr_b.keys())

    changes = {}
    novel = {}
    for key in keys:
        a, b = ctr_a[key], ctr_b[key]
        if a == 0:
            novel[key] = b
        else:
            changes[key] = float(b) / a

    print 'frequency changes'
    plt.figure()
    util.plot_pdf(changes.values(), cov_factor=0.1)
    plt.show()

    if len(novel) != 0:
        print 'novel'
        plt.figure()
        util.plot_pdf(novel.values(), cov_factor=0.1)
        plt.show()


def summarize_neighborhood(graph, seed=None, max_depth=2, nbr_samples=20, save_path=None):
    if seed is None:
        seed = random.choice(graph.neighbors.keys())
        print 'seed:', seed

    triples = set()
    explored = set()
    queue = deque()
    queue.append((seed, 0))
    while len(queue) != 0:
        entity, depth = queue.popleft()

        if depth >= max_depth:
            continue

        # loop through each available relation
        for r in graph.neighbors[entity]:
            # sample neighbors
            nbrs = graph.neighbors[entity][r]
            sampled_nbrs = util.sample_if_large(nbrs, nbr_samples, replace=False)
            num_missed = len(nbrs) - len(sampled_nbrs)

            edge_crossed = lambda target: (entity, r, target) if not inverted(r) else (target, invert(r), entity)

            # document edges crossed, and add nbrs to queue
            for nbr in sampled_nbrs:
                triples.add(edge_crossed(nbr))
                if nbr not in explored:
                    queue.append((nbr, depth + 1))

            # add "summary entity" for all entities we missed
            if num_missed > 0:
                triples.add(edge_crossed('{}_{}_{}'.format(entity, r, num_missed)))

    if save_path is not None:
        with open(save_path, 'w') as f:
            for tr in triples:
                f.write('\t'.join(tr) + '\n')

    return list(triples)


def accuracy(thresholds, examples):
    correct = 0.0
    for ex in examples:
        p = ex.score > thresholds[ex.r[0]]
        if p == ex.label:
            correct += 1
    return correct / len(examples)


def compute_best_thresholds(examples, debug=False):
    # per-relation thresholds
    ex_by_rel = util.group(examples, lambda q: q.r[0])
    thresholds = {}
    for relation, examples_r in util.verboserate(ex_by_rel.items()):
        if debug:
            print relation
        scores = [ex.score for ex in examples_r]
        labels = [ex.label for ex in examples_r]
        thresholds[relation] = util.best_threshold(scores, labels, debug)

    return thresholds


def final_evaluation(dataset_path, model_name, params_path, eval_type, eval_samples=float('inf'),
                     max_negative_samples=float('inf'), type_matching_negs=True):
    dset = parse_dataset(dataset_path)
    model = CompositionalModel(None, path_model=model_name, objective='margin')
    params = load_params(params_path, model_name)
    neg_gen = NegativeGenerator(dset.full_graph, max_negative_samples, type_matching_negs=type_matching_negs)

    queries = util.sample_if_large(dset.test, eval_samples, replace=False)

    # Define different evaluation functions
    # ----- ----- ----- ----- -----
    scores = lambda query: model.predict(params, query).ravel()

    def performance(query):
        s, r, t = query.s, query.r, query.t
        negatives = neg_gen(query, 't')
        pos_query = PathQuery(s, r, t)
        neg_query = PathQuery(s, r, negatives)

        # don't score queries with no negatives
        if len(negatives) == 0:
            query.quantile = np.nan
        else:
            query.quantile = util.average_quantile(scores(pos_query), scores(neg_query))

        query.num_candidates = len(negatives) + 1

        attributes = query.s, ','.join(query.r), query.t, str(query.quantile), str(query.num_candidates)
        return '\t'.join(attributes)

    def report(queries):
        # filter out NaNs
        queries = [q for q in queries if not np.isnan(q.quantile)]
        util.metadata('mean_quantile', np.mean([q.quantile for q in queries]))
        util.metadata('h10', np.mean([1.0 if util.rank_from_quantile(q.quantile, q.num_candidates) <= 10 else 0.0 for q in queries]))

    def average_quantile(s, p):
        negatives, positives = neg_gen(PathQuery(s, p, ''), 't', return_positives=True)
        pos_query = PathQuery(s, p, positives)
        neg_query = PathQuery(s, p, negatives)
        return util.average_quantile(scores(pos_query), scores(neg_query))

    def intermediate_aqs(query):
        s, path = query.s, query.r
        aqs = []
        for length in 1 + np.arange(len(path)):
            p = path[:length]
            aq = average_quantile(s, p)
            aqs.append(aq)

        attributes = query.s, ','.join(query.r), query.t, ','.join(str(aq) for aq in aqs)
        return '\t'.join(attributes)

    # ----- ----- ----- ----- -----

    if eval_type == 'mean_quantile':
        eval_fxn = performance
        eval_report = report
    elif eval_type == 'intermediate_aqs':
        eval_fxn = intermediate_aqs
        eval_report = lambda qs: None
    else:
        raise ValueError(eval_type)

    with open('results.tsv', 'w') as f:

        def progress(steps, elapsed):
            print '{} of {} processed ({} s)'.format(steps, len(queries), elapsed)
            util.metadata('steps', steps)
            util.metadata('gb_used', util.gb_used())
            sys.stdout.flush()
            f.flush()

        for query in util.verboserate(queries, report=progress):
            s = eval_fxn(query)
            f.write(s)
            f.write('\n')

    eval_report(queries)

    with open('queries.cpkl', 'w') as f:
        pickle.dump(queries, f)


def difference_evaluation(name):
    queries = []
    with open(join(data_directory, name + '.tsv'), 'r') as f:
        for line in util.verboserate(f):
            items = line.split('\t')
            s, r, t = items[0], tuple(items[1].split(',')), items[2]
            q = PathQuery(s, r, t)
            q.aqs = [float(s) for s in items[3].split(',')]
            queries.append(q)

    aq_deltas = defaultdict(list)
    for q in queries:
        aqs = [1.0] + q.aqs
        for i in range(1, len(aqs)):
            r = q.r[i-1]
            aq, prev_aq = aqs[i], aqs[i-1]

            if prev_aq == 1.0:
                delta = 1.0  # no ground to gain
            elif prev_aq == 0.0:
                delta = np.nan  # no ground to lose
            else:
                diff = aq - prev_aq
                if diff >= 0:
                    delta = diff / (1.0 - prev_aq)  # portion recovered
                else:
                    delta = diff / prev_aq  # portion lost

            if not np.isnan(delta):
                aq_deltas[r].append(delta)

    return pd.DataFrame({'mean(aq_diff)': dict((r, np.nanmean(deltas)) for r, deltas in aq_deltas.iteritems())})


def segmented_evaluation(file_path, categorize=None):
    queries = []
    with open(file_path, 'r') as f:
        for line in util.verboserate(f):
            items = line.split('\t')
            s, r, t = items[0], tuple(items[1].split(',')), items[2]
            q = PathQuery(s, r, t)
            quantile_str = items[3]
            q.quantile = float(quantile_str)
            q.num_candidates = int(items[4])
            queries.append(q)

    def single_relation(query):
        if len(query.r) != 1:
            return False
        r = query.r[-1]
        if inverted(r):
            return False
        return r

    # group queries
    if categorize is None:
        categorize = single_relation

    groups = util.group(queries, categorize)

    print 'computing grouped stats'
    stats = defaultdict(dict)
    for key, queries in util.verboserate(groups.iteritems()):
        scores = [q.quantile for q in queries]
        score = np.nanmean(scores)

        def inv_sigmoid(y):
            return -np.log(1.0 / y - 1)

        score2 = inv_sigmoid(score)

        total = len(scores)
        nontrivial = np.count_nonzero(~np.isnan(scores))

        stats[key] = {'score': score, 'score2': score2, 'total_eval': total, 'nontrivial_eval': nontrivial}

    stats.pop(False, None)
    return pd.DataFrame(stats).transpose()


def load_socher_test(test_set_path):
    examples = []
    with open(join(data_directory, test_set_path), 'r') as f:
        for line in util.verboserate(f):
            items = line.split()
            s, r, t, label = items[0], tuple(items[1].split(',')), items[2], items[3]
            ex = PathQuery(s, r, t)

            if label == '1':
                ex.label = True
            elif label == '-1':
                ex.label = False
            else:
                raise ValueError(label)
            examples.append(ex)
    return examples


def find_correlations(df, graph):
    df_rel = pd.DataFrame(graph.relation_stats()).transpose()
    df = pd.concat((df_rel, df), axis=1, join='inner')

    print 'creating scatter plot'
    plt.figure()
    pd.tools.plotting.scatter_matrix(df)
    plt.tight_layout()
    plt.show()

    return df


def satisfying_pairs(p, graph):
    pairs = set()

    sources = graph.type_matching_entities(p, 's')

    for s in util.verboserate(sources):
        if len(p) == 1:
            for t in graph.neighbors[s][p[0]]:
                pairs.add((s, t))
        else:
            for t in graph.walk_all(s, p):
                pairs.add((s, t))
    return pairs


def path_correlation(path, relation, graph):
    # find all pairs satisfying path
    path_pairs = satisfying_pairs(path, graph)
    rel_pairs = satisfying_pairs((relation,), graph)

    ni = float(len(path_pairs & rel_pairs))
    npaths = float(len(path_pairs))
    nrels = float(len(rel_pairs))

    if npaths != 0:
        precision = ni / npaths
    else:
        precision = np.nan

    recall = ni / nrels

    return precision, recall


def path_angle(path, relation, params, model='bilinear'):
    if model != 'bilinear':
        raise NotImplementedError(model)

    w_p = 1.0
    for r in path:
        w = params[('r', r)]
        w_p = w.dot(w_p)

    w_r = params[('r', relation)]

    # L2 norm
    inner = lambda a, b: np.sum(a * b)
    norm = lambda a: np.sqrt(inner(a, a))
    angle = np.arccos(inner(w_r, w_p) / (norm(w_r) * norm(w_p)))

    return angle


class AccuracyObserver(Observer):

    def __init__(self, dset_path, eval_samples=float('inf'), report_wait=30):
        self.examples = load_socher_test(dset_path)
        self.eval_samples = eval_samples
        self.report_wait = report_wait

    def observe(self, maximizer, thresholds=None):
        if (maximizer.steps + 1) % self.report_wait != 0:
            return None

        samples = util.sample_if_large(self.examples, self.eval_samples, replace=True)

        # score
        samples = copy.deepcopy(samples)
        for ex in samples:
            try:
                ex.score = maximizer.objective.predict(maximizer.params, ex).ravel()[0]
            except KeyError:
                print 'out of vocab'
                ex.score = float('inf')

        if thresholds is None:
            thresholds = compute_best_thresholds(samples)
        acc = accuracy(thresholds, samples)

        return {('accuracy', 'test'): acc}


class RankObserver(Observer):
    """
    Approximately computes mean rank over a dataset of PathQuery's.
    Note that this does NOT necessarily reflect the real pos/neg proportions
    when computing exact mean rank.
    """

    def __init__(self, dsets, full_graph, eval_samples, max_negative_samples, report_wait, type_matching_negs=True):
        self.dsets = dsets
        self.full_graph = full_graph
        self.eval_samples = eval_samples
        self.report_wait = report_wait
        self.neg_generator = NegativeGenerator(full_graph, max_negative_samples, type_matching_negs=type_matching_negs)

    def observe(self, maximizer):
        if (maximizer.steps + 1) % self.report_wait == 0:

            # print 'train example'
            # self.examine_ranking(maximizer, random.choice(maximizer.train))
            # print 'dev example'
            # self.examine_ranking(maximizer, random.choice(maximizer.dev))

            metrics = {}
            for name, dset in self.dsets.iteritems():
                if name == 'train' and self.eval_samples == float('inf'):
                    continue
                metrics[('mean_rank', name)] = self.mean_rank(maximizer, dset)
            return metrics

        return None

    @util.profile
    def mean_rank(self, maximizer, dset):
        sample = util.sample_if_large(dset, self.eval_samples)
        ranks = [self.rank(maximizer, ex) for ex in util.verboserate(sample)]
        return np.nanmean(ranks)

    def rank(self, maximizer, ex):
        samples, scores, ranks = self.predict(maximizer, ex)
        return ranks[0]

    @util.profile
    def predict(self, maximizer, ex):
        samples = self.neg_generator(ex, 't')
        samples.insert(0, ex.t)  # insert positive at front

        scores = maximizer.objective.predict(maximizer.params, PathQuery(ex.s, ex.r, samples)).ravel()
        assert len(scores.shape) == 1

        ranks = util.ranks(scores, ascending=False)
        return samples, scores, ranks

    def examine_ranking(self, maximizer, ex):
        samples, scores, ranks = self.predict(maximizer, ex)
        examine_ranking(ex, samples, scores, ranks)