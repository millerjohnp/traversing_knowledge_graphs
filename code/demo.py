from optimize import *
from diagnostics import *
import configs
import argparse
from data import *
import copy


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('dataset_path')
parser.add_argument('-i', '--initial_params', default=None)
parser.add_argument('-glove', '--glove_vectors', default=None)

args = parser.parse_args()

# load pre-set configuration from configs module
config = getattr(configs, args.config)
config['dataset_path'] = args.dataset_path
config['params_path'] = args.initial_params
config['glove_path'] = args.glove_vectors

# load all configs into local namespace
for var, val in config.iteritems():
    exec("{0} = config['{0}']".format(var))
    util.metadata(var, val)  # this logs parameters to a metadata file.


def print_header(msg):
    print
    print msg.upper()
    print '=====' * 5
    print


# define training procedure
def build_trainer(train, test, max_steps, step_size, init_params=None):
    # negative triple generator for training
    triples = [(q.s, str(q.r[0]), q.t) for q in train if len(q.r) == 1]
    train_graph = Graph(triples)
    train_neg_gen = NegativeGenerator(train_graph, max_negative_samples_train,
                                      positive_branch_factor, type_matching_negs
                                      )

    # specify the objective to maximize
    objective = CompositionalModel(train_neg_gen, path_model=path_model,
                                   objective='margin')

    # initialize params if not already initialized
    if init_params is None:
        init_params = objective.init_params(
            dset.entity_list, dset.relations_list, wvec_dim, model=path_model,
            hidden_dim=hidden_dim, init_scale=init_scale, glove_path=glove_path)

    save_wait = 1000  # save parameters after this many steps
    eval_samples = 200  # number of examples to compute objective on

    # define Observers
    observers = [NormObserver(report_wait), SpeedObserver(report_wait),
                 ObjectiveObserver(eval_samples, report_wait)]

    # this Observer computes the mean rank on each split
    rank_observer = RankObserver({'train': train, 'test': test},
                                 dset.full_graph, eval_samples,
                                 max_negative_samples_eval, report_wait,
                                 type_matching_negs=True)
    observers.append(rank_observer)

    # define Controllers
    controllers = [BasicController(report_wait, save_wait, max_steps),
                   DeltaClipper(), AdaGrad(), UnitNorm()]

    trainer = OnlineMaximizer(
        train, test, objective, l2_reg=l2_reg, approx_reg=True,
        batch_size=batch_size, step_size=step_size, init_params=init_params,
        controllers=controllers, observers=observers)

    return trainer


dset = parse_dataset(dataset_path, dev_mode=False, maximum_examples=100)

warm_start = params_path is not None
if warm_start:
    print 'loading warm start params...'
    init_params = load_params(params_path, path_model)
else:
    init_params = None

print_header('single-edge training')

# train the model on single edges
one_hop_only = lambda queries: [q for q in queries if len(q.r) == 1]
trainer0 = build_trainer(
    one_hop_only(dset.train), one_hop_only(dset.test),
    max_steps_single, step_size_single, init_params
)
params0 = trainer0.maximize()
params_single = copy.deepcopy(params0)

print_header('path training')

# train the model on all edges, with warm start from single-edge model
trainer = build_trainer(dset.train, dset.test, max_steps_path,
                        step_size_path, params0)
params_comp = trainer.maximize()

print_header('evaluation')


def report(queries, model, neg_gen, params):
    scores = lambda query: model.predict(params, query).ravel()

    def compute_quantile(query):
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

    for query in util.verboserate(queries):
        compute_quantile(query)

    # filter out NaNs
    queries = [q for q in queries if not np.isnan(q.quantile)]
    mean_quantile = np.mean([q.quantile for q in queries])
    hits_at_10 = np.mean([1.0 if util.rank_from_quantile(q.quantile, q.num_candidates) <= 10 else 0.0 for q in queries])

    print 'mean_quantile:', mean_quantile
    print 'h10', hits_at_10

    return mean_quantile, hits_at_10

# used for all evaluations
neg_gen = NegativeGenerator(dset.full_graph, float('inf'), type_matching_negs=True)

print_header('path query evaluation')

print '--Single-edge trained model--'
mq, h10 = report(dset.test, trainer0.objective, neg_gen, params_single)
util.metadata(('path_queries', 'SINGLE', 'mq'), mq)
util.metadata(('path_queries', 'SINGLE', 'h10'), h10)
print

print '--Compositional trained model--'
mq, h10 = report(dset.test, trainer.objective, neg_gen, params_comp)
util.metadata(('path_queries', 'COMP', 'mq'), mq)
util.metadata(('path_queries', 'COMP', 'h10'), h10)
print

print_header('single edge evaluation')
print '--Single-edge trained model--'
mq, h10 = report(one_hop_only(dset.test), trainer0.objective, neg_gen, params_single)
util.metadata(('single_edges', 'SINGLE', 'mq'), mq)
util.metadata(('single_edges', 'SINGLE', 'h10'), h10)
print

print '--Compositional trained model--'
mq, h10 = report(one_hop_only(dset.test), trainer.objective, neg_gen, params_comp)
util.metadata(('single_edges', 'COMP', 'mq'), mq)
util.metadata(('single_edges', 'COMP', 'h10'), h10)
