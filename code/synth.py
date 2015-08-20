import platform
import random
import data
from data import PathQuery, Graph
import cPickle as pickle
import util
import multiprocessing as mp

__author__ = 'Kelvin Gu'


def augment_dataset(train_triples, dev_triples, add_paths=False, max_path_length=8):

    train_graph = Graph(train_triples)
    full_graph = Graph(train_triples + dev_triples)

    # start with original edges in the training and dev set
    train = PathQuery.from_triples(train_triples)
    dev = PathQuery.from_triples(dev_triples)
    test = []  # empty for now

    if add_paths:
        print 'adding paths'
        # number of paths to augment existing triples with
        num_augment = lambda triples: len(triples) * (max_path_length - 1)

        # augment with paths
        train.extend(sample_paths(train_graph, 3*num_augment(train_triples), max_path_length))  # previously 980000
        dev.extend(sample_paths(full_graph, num_augment(dev_triples), max_path_length))  # previously 35000

    # make unique, and eliminate train dev overlap
    print 'before: train {}, dev {}'.format(len(train), len(dev))
    train, dev = set(train), set(dev)
    dev -= train

    # remove trivial queries (queries that type match all entities)
    trivial_train_paths = set(get_trivial_path_queries(train_graph, train))
    trivial_dev_paths = set(get_trivial_path_queries(train_graph, dev))

    train -= trivial_train_paths
    dev -= trivial_dev_paths

    train, dev = list(train), list(dev)
    random.shuffle(train)
    random.shuffle(dev)

    print 'after: train {}, dev {}'.format(len(train), len(dev))

    if platform.system() != 'Darwin':
        # save generated datasets
        print 'saving datasets'
        dsets = {'train': train, 'dev': dev, 'test': test}
        for name, dset in dsets.iteritems():
            with open('{}.cpkl'.format(name), 'w') as f:
                pickle.dump(dset, f)
        print 'done'

    return train, dev, test, train_graph, full_graph


def sample_paths(graph, num_paths, max_path_length):
    paths = []
    for k in util.verboserate(range(num_paths)):
        length = random.randint(2, max_path_length)  # don't include length 1
        paths.append(graph.random_path_query(length))
    return paths


def get_trivial_path_queries(graph, queries):
    print "Filtering queries that type match all entities"

    in_query_queue = mp.JoinableQueue()
    results_queue = mp.Queue()

    for query in queries:
        in_query_queue.put(query)

    def worker():
        while True:
            query = in_query_queue.get()
            if isinstance(query, PathQuery) and graph.is_trivial_query(query.s, query.r):
                results_queue.put(query)
            in_query_queue.task_done()

    # launch jobs
    jobs = list()
    for i in xrange(mp.cpu_count()):
        p = mp.Process(target=worker)
        p.start()
        jobs.append(p)

    in_query_queue.join()

    trivial_queries = []
    while not results_queue.empty():
        trivial_queries.append(results_queue.get())

    print "Number of trivial queries: ", len(trivial_queries)
    return trivial_queries


def group_queries_by_difficulty(train_graph, full_graph, queries, existence=True, epsilon=5e-1):
    print "Filtering queries contained in train graph"
    easy_queries = []
    hard_queries = []
    for query in util.verboserate(queries):
        if existence:

            if isinstance(query, PathQuery) or isinstance(query, data.PathQuery):
                easy = query.t in train_graph.walk_all(query.s, query.r)
            else:
                raise TypeError(type(query))

            if easy:
                easy_queries.append(query)
            else:
                hard_queries.append(query)
        else:

            mc_estimates = train_graph.random_walk_probs(query.s, query.r)
            if query.t in mc_estimates:
                approx = mc_estimates[query.t]
            else:
                approx = 0.
            true = full_graph.random_walk_probs(query.s, query.r)[query.t]
            if abs(true - approx) < epsilon:
                easy_queries.append(query)
            else:
                hard_queries.append(query)

    print "Number of easy queries: ", len(easy_queries)
    print "Number of hard queries: ", len(hard_queries)
    return easy_queries, hard_queries
