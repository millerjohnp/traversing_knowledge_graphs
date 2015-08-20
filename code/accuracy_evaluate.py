
# coding: utf-8

# In[ ]:

import scriptinit
import util
from data import *
from optimize import *
import diagnostics as dns
from os.path import join
from composition import *

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse
from os.path import join
import cPickle


# In[ ]:

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('params')

if util.in_ipython():
    args = parser.parse_args(['freebase_socher', 'bilinear', 'compositional_wvec_bilinear_freebase_socher_0xe7d4cf_params.cpkl'])
else:
    args = parser.parse_args()

util.metadata('dataset', args.dataset)
util.metadata('model', args.model)
util.metadata('params', args.params)
util.metadata('split', 'test')
    
model = CompositionalModel(None, path_model=args.model, objective='margin')
params = load_params(args.params, args.model)

dev = dns.load_socher_test(join(args.dataset, 'dev'))
test = dns.load_socher_test(join(args.dataset, 'test'))

def score(samples):
    for ex in samples:
        try:
            ex.score = model.predict(params, ex).ravel()[0]
        except KeyError:
            print 'out of vocab'
            ex.score = float('inf')

score(dev)
score(test)

thresholds = dns.compute_best_thresholds(dev)
util.metadata('accuracy', dns.accuracy(thresholds, test))

