import theano
import theano.tensor as T
import util


#########################
# PATH QUERY MODELS     #
#########################
@util.memoize
def get_path_query_model(model, objective=None):
    if model == 'bilinear':
        assert objective is not None
        return bilinear_model(objective)
    elif model == 'transE':
        return transE_model()
    elif model == 'NTN':
        return neural_tensor_network()
    elif model == 'bilinear_diag':
        return bilinear_diag()
    else:
        raise NotImplementedError()


def bilinear_diag():
    # W should be a matrix with the diagonal matrix for the i-th relation
    # stored as the i-th column of W
    X_s = T.matrix('X_s')
    W = T.matrix('W')
    X_t = T.matrix('X_t')

    diags = W[:, :, None].transpose(1, 0, 2)

    results, updates = theano.scan(fn=lambda diag, v: diag * v,
                                   outputs_info=X_s, sequences=[diags])

    # # score is always a column vector
    score = X_t.T.dot(results[-1]).reshape((-1, 1))

    # assumes the first entry is the correct score
    margin = score[0] - score[1:]
    cost = T.min(T.concatenate((T.ones_like(margin), margin), axis=1), axis=1).mean()

    gX_s, gW, gX_t = T.grad(cost, [X_s, W, X_t])

    print 'compiling bilinear diag fprop'
    fprop = theano.function([X_s, W, X_t], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_input = True

    print 'compiling bilinear diag bprop'
    bprop = theano.function([X_s, W, X_t], [gX_s, gW, gX_t],
                            name='bprop', mode='FAST_COMPILE')
    bprop.trust_input = True

    print 'compiling bilinear diag score'
    score = theano.function(inputs=[X_s, W, X_t], outputs=score,
                            name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'score': score, 'fprop': fprop, 'bprop': bprop}


def transE_model():
    '''
        Note X_S is a column and X_T is a matrix so that broadcasting occurs
        across the columns of X_T (this allows batching X_T with negatives,
        for example.
    '''
    # construct theano expression graph
    X_s = T.col('X_s')
    W = T.matrix('W')
    X_t = T.matrix('X_t')

    rels = W[:, :, None].transpose(1, 0, 2)

    # Computes x_{r_1} + x_{r_{2}} + ... + x_{r_n} - X_{t}
    results, updates = theano.scan(fn=lambda rel, v: rel + v,
                                   outputs_info=-X_t, sequences=[rels])

    # score is always a column vector
    score = T.sum((X_s + results[-1]) ** 2, axis=0).reshape((-1, 1))

    margins = 1. + score[0] - score[1:]

    # zero out negative entries
    pos_parts = margins * (margins > 0)

    # we are using online Maximizer, so the objective is negated
    cost = -pos_parts.mean()

    gX_s, gW, gX_t = T.grad(cost, [X_s, W, X_t])

    print 'Compiling TransE score'
    # return negative score since this is a ranking
    score = theano.function([X_s, W, X_t], -score, name='transE Score',
                            mode='FAST_RUN')
    score.trust_input = True

    print 'Compiling TransE fprop'
    fprop = theano.function([X_s, W, X_t], cost, name='transE fprop',
                            mode='FAST_RUN')
    fprop.trust_input = True

    print 'Compiling TransE bprop'
    bprop = theano.function([X_s, W, X_t],
                            outputs=[gX_s, gW, gX_t],
                            name='transE bprop', mode='FAST_RUN')
    bprop.trust_input = True

    return {'score': score, 'fprop': fprop, 'bprop': bprop}


# Note: Model assumes we only use corrupted target entities with the first
# target entity the true entity
def neural_tensor_network():
    # tensor params
    subj = T.col('e_1')
    targets = T.matrix('e_2')
    W = T.tensor3('W')

    # neural net params
    u = T.col('u')
    V = T.matrix('V')
    b = T.col('b')

    # tensor
    h = subj.T.dot(W).dot(targets)

    # neural net
    d = subj.shape[0]
    V_subj = V[:, :d].dot(subj)
    V_targ = V[:, d:].dot(targets)

    activations = T.tanh(h + V_subj + V_targ + b)
    score = u.T.dot(activations).reshape((-1, 1))

    margins = score[0] - score[1:]
    cost = T.min(T.concatenate((T.ones_like(margins), margins), axis=1), axis=1).mean()

    gsubj, gtargets, gW, gu, gV, gb = T.grad(cost, [subj, targets, W, u, V, b])

    print 'Compiling NTN score'
    score = theano.function([subj, W, targets, u, V, b], score, name='NTN Score',
                            mode='FAST_RUN')

    print 'Compiling NTN fprop'
    fprop = theano.function([subj, W, targets, u, V, b], cost, name='NTN fprop',
                            mode='FAST_RUN')

    print 'Compiling NTN bprop'
    bprop = theano.function([subj, W, targets, u, V, b],
                            outputs=[gsubj, gW, gtargets, gu, gV, gb],
                            name='NTN bprop', mode='FAST_RUN')

    return {'score': score, 'fprop': fprop, 'bprop': bprop}


def bilinear_model(objective):
    # construct theano expression graph
    X_s = T.matrix('X_s')
    W = T.tensor3('W')
    X_t = T.matrix('X_t')

    results, updates = theano.scan(fn=lambda W, v: W.dot(v),
                                   outputs_info=X_s, sequences=[W])

    # score is always a column vector
    score = X_t.T.dot(results[-1]).reshape((-1, 1))

    # # SUM OF MARGINS #####
    # # assumes the first entry is the correct score
    # margin = score[0] - score[1:]
    # cost = T.min(T.concatenate((T.ones_like(margin), margin), axis=1), axis=1).mean()
    # ######################

    ## MAX OF MARGINS ####
    # assumes the first entry is the correct score
    max_margin = score[0] - T.max(score[1:])
    cost = T.min(T.concatenate((T.ones_like(max_margin), max_margin)))
    ######################

    gX_s, gW, gX_t = T.grad(cost, [X_s, W, X_t])

    print 'compiling bilinear fprop'
    fprop = theano.function([X_s, W, X_t], cost, name='fprop', mode='FAST_RUN')
    fprop.trust_input = True

    print 'compiling bilinear bprop'
    bprop = theano.function([X_s, W, X_t], [gX_s, gW, gX_t],
                            name='bprop', mode='FAST_RUN')
    bprop.trust_input = True

    print 'compiling bilinear score'
    score = theano.function(inputs=[X_s, W, X_t], outputs=score,
                            name='score', mode='FAST_RUN')
    score.trust_input = True

    return {'score': score, 'fprop': fprop, 'bprop': bprop}
