
# coding: utf-8

# In[ ]:

import scriptinit
import clb
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
from os import path
import cPickle as pickle


# In[ ]:

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
parser.add_argument('relation')
parser.add_argument('baseline')
parser.add_argument('compositional')

# good one for illustration
# path: ('parents', 'location')
# relation: 'place_of_birth'

if util.in_ipython():
    args = parser.parse_args(['freebase_socher', 'bilinear', 'nationality', 'bilinear_freebase_socher_0x77cdc2_params.cpkl', 'compositional_params_freebase'])
else:
    args = parser.parse_args()

dset = parse_dataset(args.dataset)
baseline = load_params(args.baseline, args.model)
compositional = load_params(args.compositional, args.model)

def compare_angles(p):
    b_angle = dns.path_angle(p, args.relation, baseline)
    c_angle = dns.path_angle(p, args.relation, compositional)
    return (c_angle - b_angle) / b_angle

relations = dset.full_graph.relation_args.keys()
relations += [invert(r) for r in relations]
print relations


# In[ ]:

pts = []
for r1 in relations:
    for r2 in relations:
        p = (r1, r2)
        print p
        delta = compare_angles(p)
        score, _ = dns.path_correlation(p, args.relation, dset.full_graph)
        pts.append((p, score, delta))
        
        with open('points.cpkl', 'w') as f:
            pickle.dump(pts, f)


# In[ ]:

# with open('points.cpkl', 'r') as f:
#     pts = pickle.load(f)
# paths, scores, deltas = zip(*pts)
# plt.scatter(scores, deltas)

