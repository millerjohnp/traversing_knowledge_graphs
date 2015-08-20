
# coding: utf-8

# In[ ]:

import scriptinit
from data import *
from synth import *
import argparse
import os


# In[ ]:

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path')

if util.in_ipython():
    args = parser.parse_args(['wordnet_clean'])
else:
    args = parser.parse_args()


# In[ ]:

def save_datasets(train, dev, test, dev_easy, dev_hard, test_easy, test_hard, directory):
    # save generated datasets
    print 'saving datasets'
    os.mkdir(directory)

    dsets = {'train': train, 'dev': dev, 'test': test, 'dev_approx': dev_easy, 'dev_gen': dev_hard,
            'test_approx': test_easy, 'test_gen': test_hard}
    for name, dset in dsets.iteritems():
        with open(os.path.join(directory, '{}'.format(name)), 'w') as f:
            for triple in dset:
                e1, r, e2 = triple.s, triple.r, triple.t
                f.write(e1 + '\t' + ','.join(r) + '\t' + e2 + '\n')
    print 'done'
        


# In[ ]:

MAX_PATH_LENGTH = 5

dataset_path = args.dataset_path

def generate_final_dataset():
    entity_list, relation_list, train_triples, dev_triples, test_triples = load_dataset(dataset_path)
    
    # full set of triples
    triples = train_triples + dev_triples
    # augment the relation_list with the inverse relations
    relation_list.extend([invert(r) for r in relation_list])
    
    train_graph = Graph(train_triples)
    dev_graph = Graph(train_triples + dev_triples)
    test_graph = Graph(train_triples + test_triples)

    # start with original edges in the training, dev, and test sets
    train = PathQuery.from_triples(train_triples)
    dev = PathQuery.from_triples(dev_triples)
    test = PathQuery.from_triples(test_triples)
    
    # set random seed for reproducibility
    random.seed(0)
    np.random.seed(0)
    
    # add paths
    print 'Adding paths!'
    num_augment = lambda triples: len(triples) * (MAX_PATH_LENGTH - 1)

    train.extend(sample_paths(train_graph, 5*num_augment(train_triples), MAX_PATH_LENGTH))  # previously 980000
    dev.extend(sample_paths(dev_graph, num_augment(dev_triples), MAX_PATH_LENGTH))  # previously 35000
    
    # make unique, and eliminate train dev overlap
    print 'before: train {}, dev {}'.format(len(train), len(dev))
    train, dev = set(train), set(dev)
    dev -= train
    train, dev = list(train), list(dev)
    random.shuffle(train)
    random.shuffle(dev)

    print 'after: train {}, dev {}'.format(len(train), len(dev))

    print 'Generating test dataset...'

    test.extend(sample_paths(test_graph, num_augment(test_triples), MAX_PATH_LENGTH))

    # ensure the test set is unique
    print "before: test {} ".format(len(test))
    train, dev, test = set(train), set(dev), set(test)
    test -= train
    test -= dev

    train, dev, test = list(train), list(dev), list(test)

    random.shuffle(test)

    print 'after: test {}'.format(len(test))

    # get dev paths by type
    dev_easy, dev_hard = group_queries_by_difficulty(train_graph, dev_graph, dev, existence=True)

    # get train paths by type
    test_easy, test_hard = group_queries_by_difficulty(train_graph, test_graph, test, existence=True)

    save_datasets(train, dev, test, dev_easy, dev_hard, test_easy, test_hard, dataset_path + '_paths_test')


generate_final_dataset()


# In[ ]:




# In[ ]:



