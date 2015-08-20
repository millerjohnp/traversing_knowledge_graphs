import abc
from collections import MutableMapping, defaultdict
from Queue import Queue
import cPickle as pickle
import os
import threading
import types
import copy
import random
import math
import sys
from datetime import datetime
import time
import numpy as np
import resource

import util


__author__ = 'Kelvin Gu'


class SparseVector(MutableMapping):
    # rather than inheriting from dict, SparseVector contains a dict
    # this is useful in some situations, e.g. if we'd like multiple SparseVectors that are all a view of the same
    # underlying dict

    # METHODS THAT USE d (should be the only ones that access it)
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    def __init__(self, d=None):
        if d is None:
            d = {}

        assert isinstance(d, dict)
        self.__d = d

    def __getitem__(self, key):
        return self.__d.__getitem__(key)

    def __setitem__(self, key, value):
        # value must be a numpy array or float
        assert isinstance(value, np.ndarray) or isinstance(value, float)
        self.__d.__setitem__(key, value)
        return self

    def __delitem__(self, key):
        return self.__d.__delitem__(key)

    def __iter__(self):
        # if we don't implement this, iter() will see that we did implement __getitem__
        # and mistakenly assume that this is a sequence
        # it will then try to call self.__getitem__(n) for n = 0, 1, 2, ...
        return iter(self.__d)

    def __len__(self):
        return self.__d.__len__()

    def __contains__(self, item):
        # if we don't implement this, then Python will iterate over self.__iter__() checking for the item
        # this is inefficient
        return item in self.__d

    def as_dict(self):
        return self.__d

    def __str__(self):
        return str(self.__d)

    def __repr__(self):
        return repr(self.__d)

    def plot(self):
        for name, val in self.__d.iteritems():
            print name
            util.matshow(val)

    # UNARY OPERATIONS
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    def select(self, keys):
        """
        Return a new SparseVector with the selected entries.
        """
        new = SparseVector()
        for key in keys:
            val = self.get(key)
            if val is not None:
                new[key] = val

        return new

    def remove(self, key):
        return self.pop(key, None)

    def iapply(self, f):
        """
        Apply the function f(key, val) -> new value, in place
        """
        for key, val in self.iteritems():
            self[key] = f(key, val)

        return self

    def _copy_op(self, op):
        """
        Copy self and perform op(SparseVector) -> SparseVector, on copy
        """
        new = copy.deepcopy(self)
        op(new)
        return new

    def apply(self, f):
        def op(x):
            x.iapply(f)

        return self._copy_op(op)

    def norm2(self):
        sq_sum = 0
        for val in self.itervalues():
            sq_sum += np.linalg.norm(val) ** 2

        return math.sqrt(sq_sum)

    # BINARY OPERATIONS
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    def add(self, key, val):
        """
        This method is here for efficiency
        It assumes that if an entry is missing, its value is 0
        """
        orig_val = self.get(key)
        if orig_val is None:
            self[key] = val
        else:
            orig_val += val
            # this is necessary if orig_val and val are floats
            self[key] = orig_val

    def __iadd__(self, other):
        for key, other_val in other.iteritems():
            val = self.get(key)
            if val is None:
                self[key] = other_val
            else:
                val += other_val
                # this is necessary if val is a float
                self[key] = val

        return self

    def __imul__(self, other):
        isfloat = isinstance(other, float)

        # we assume that LHS is sparser
        to_remove = []
        for key, val in self.iteritems():
            # if other is a scalar, just do scalar multiplication
            other_val = other if isfloat else other.get(key)

            if other_val is None:
                to_remove.append(key)
            else:
                val *= other_val
                # this is necessary if val is a float
                self[key] = val

        for key in to_remove:
            self.pop(key)

        return self

    def __add__(self, other):
        def op(x):
            x += other

        return self._copy_op(op)

    def __mul__(self, other):
        def op(x):
            x *= other

        return self._copy_op(op)

    __radd__ = __add__
    __rmul__ = __mul__

    def __eq__(self, other):
        # check that other is SparseVector
        if not isinstance(other, SparseVector):
            return False

        # checking self.d == other.d does not work, because the == operator on arrays produces another array

        # check matching keys
        if self.keys() != other.keys():
            return False

        # check matching values
        for key in self.iterkeys():
            v1 = self[key]
            v2 = other[key]
            if (v1 != v2).any():
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def dot(self, other):
        prod = self * other
        sum = 0.0
        for val in prod.itervalues():
            sum += np.sum(val)
        return sum

    # NON-INSTANCE METHODS
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    @classmethod
    def random(cls, x, sigma=1.0):
        """
        Generate a random vector with the same keys and dimensions as x
        """
        rand = cls()
        for key, val in x.iteritems():
            if isinstance(val, float):
                d = random.normalvariate(0, sigma)
            else:
                d = np.random.normal(scale=sigma, size=x[key].shape)
            rand[key] = d
        return rand


class Initialized(SparseVector):
    # overrides __getitem__, leaving everything else the same

    def __init__(self, sparse_vector, init_fxn):
        # note that we are sharing the same underlying dict with sparse_vector
        # modifying this object will modify the original sparse_vector!
        SparseVector.__init__(self, sparse_vector.as_dict())
        self.init_fxn = init_fxn

    # TODO: WARNING, one possibly unintended consequence...
    # suppose that you have an empty vector, params
    # when you do params + grad, you would expect to just get grad
    # however, params will initialize its entries before adding to grad, so you won't just get grad

    def __getitem__(self, key):
        try:
            return SparseVector.__getitem__(self, key)
        except KeyError:
            val = self.init_fxn(key)
            if val is None:
                raise KeyError('init fxn could not handle {}'.format(key))
            self[key] = val
            return val


class Remapped(object):
    # provides a very restricted view of a SparseVector, where the keys are remapped according to remap_fxn
    # this wrapper makes it pretty hard to accidentally modify the underlying SparseVector

    def __init__(self, sparse_vector, remap_fxn):
        self.sparse_vector = sparse_vector
        self.remap_fxn = remap_fxn

    def __getitem__(self, key):
        return self.sparse_vector[self.remap_fxn(key)]


class OnlineFunction(object):
    def value(self, params, ex):
        """
        Return a float
        """
        raise NotImplementedError()

    def gradient(self, params, ex):
        """
        Return a SparseVector
        """
        raise NotImplementedError()

    @staticmethod
    def implement(value, gradient):
        implem = OnlineFunction()

        value_self = lambda self, params, ex: value(params, ex)
        gradient_self = lambda self, params, ex: gradient(params, ex)

        # bind these to implem
        implem.value = types.MethodType(value_self, implem)
        implem.gradient = types.MethodType(gradient_self, implem)

        return implem


class SigmoidOnlineFunction(OnlineFunction):
    """
    Use this if you just want to wrap an extra sigmoid on top of some other OnlineFunction
    """

    def __init__(self, score_fxn):
        self.score_fxn = score_fxn

    def value(self, params, ex):
        return util.sigmoid(self.score_fxn.value(params, ex))

    def gradient(self, params, ex):
        inner_val = self.score_fxn.value(params, ex)
        outer_deriv = util.sigmoid_derivative(inner_val)
        inner_grad = self.score_fxn.gradient(params, ex)
        grad = inner_grad * outer_deriv
        return grad


class BernoulliObjective(OnlineFunction):
    def __init__(self, prob_fxn):
        # prob_fxn.value(params, ex) should return a probability for ex, given params
        self.prob_fxn = prob_fxn

    def value(self, params, ex):
        w = ex.weight
        y = ex.score
        f = self.prob_fxn.value(params, ex)

        assert 0.0 <= f <= 1.0

        max_float = sys.float_info.max

        # these cannot be -infinity, because -infinity * 0 = NaN
        # note that y or (1 - y) can be 0
        if f != 0.0:
            log_f = math.log(f)
        else:
            log_f = -max_float

        if f != 1.0:
            log_1_minus_f = math.log(1 - f)
        else:
            log_1_minus_f = -max_float

        ll = w * (y * log_f + (1 - y) * log_1_minus_f)

        # this ensures that the sum LL isn't -infinity:
        # if we didn't do this, one bad example could make LL -infinity
        # the effect of that bad example is now limited, allowing us to see
        # the change in the ll of other reasonable examples over time

        # assume at most 1 million training examples
        max_train = 1000000
        min_value = -max_float / max_train
        ll = max(min_value, ll)

        return ll

    def gradient(self, params, ex):
        w = ex.weight
        y = ex.score
        g = self.prob_fxn.gradient(params, ex)
        f = self.prob_fxn.value(params, ex)

        return w * g * ((y - f) / (f - f ** 2))


# note that the squared error is negated, because it is to be maximized
class SquaredObjective(OnlineFunction):
    def __init__(self, score_fxn):
        self.score_fxn = score_fxn

    def value(self, params, ex):
        w = ex.weight
        y = ex.score
        y_hat = self.score_fxn.value(params, ex)
        return -w * (y - y_hat) ** 2

    def gradient(self, params, ex):
        w = ex.weight
        y = ex.score
        y_hat = self.score_fxn.value(params, ex)
        return w * 2 * (y - y_hat) * self.score_fxn.gradient(params, ex)


class OnlineMaximizer(object):
    def __init__(self, train, dev, objective, batch_size=1, step_size=0.1, num_threads=1,
                 l1_reg=0.0, l2_reg=0.0, approx_reg=False,
                 init_params=None, init_fxn=lambda x: None, freeze_params=None,
                 controllers=None, observers=None):

        # CORE INPUTS
        # ----- ----- ----- ----- -----
        self.train = train
        self.dev = dev
        self.objective = objective

        # STEP CONTROL
        # ----- ----- ----- ----- -----
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_threads = num_threads
        self.halt = False

        # REGULARIZATION
        # ----- ----- ----- ----- -----
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.approx_reg = approx_reg

        # PARAMETER INIT / FREEZING
        # ----- ----- ----- ----- -----
        if freeze_params is None:
            freeze_params = set()

        if init_params is None:
            init_params = SparseVector()

        for param_name in freeze_params:
            # if you freeze a param, you must initialize it
            assert param_name in init_params
            # WARNING: if this doesn't hold, the parameter will NOT be properly frozen!

        # wrap parameters in initializer: if a parameter is not available, initialize it
        self.params = Initialized(init_params, init_fxn)
        self.freeze_params = freeze_params

        # TRACKING
        # ----- ----- ----- ----- -----
        if observers is None:
            observers = []
        self.observers = observers
        self.history = defaultdict(lambda: (list(), list()))

        # CONTROLLERS
        # ----- ----- ----- ----- -----
        if controllers is None:
            controllers = [BasicController()]
        self.controllers = controllers

    def track(self):
        report = []
        for observer in self.observers:
            metrics = observer.observe(self)
            if metrics is None:
                continue
            for name, val in metrics.iteritems():
                timestamps, values = self.history[name]
                timestamps.append(self.steps)
                values.append(val)

                util.metadata(name, val)
                report.append((name, val))

        if len(report) > 0:
            print ', '.join(['{}: {:.3f}'.format('.'.join(name), val) for name, val in report])
            with open('history.cpkl', 'w') as f:
                pickle.dump(dict(self.history), f)

    @util.profile
    def maximize(self):
        print 'mini-batch gd: examples = {}, batch size = {}'.format(len(self.train), self.batch_size)

        # these are for multithreading
        q_in = Queue()
        q_out = Queue()

        def worker():
            while True:
                ex = q_in.get()
                q_out.put(self.objective.gradient(self.params, ex))
                q_in.task_done()

        # launch workers
        for i in range(self.num_threads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()

        # no. of mini-batch steps taken
        self.steps = 0
        while True:
            # form fresh batches
            train_copy = list(self.train)
            random.shuffle(train_copy)
            batches = list(util.chunks(train_copy, self.batch_size))

            for batch in batches:
                grad = SparseVector()

                if self.num_threads == 1:
                    for ex in batch:
                        grad_ex = self.objective.gradient(self.params, ex)
                        grad += grad_ex
                else:
                    # WARNING: this is only safe if examples in the batch are mutually exclusive
                    for ex in batch:
                        q_in.put(ex)
                    q_in.join()
                    while not q_out.empty():
                        grad += q_out.get()

                for frozen in self.freeze_params:
                    grad.remove(frozen)

                # normalize by batch size
                grad *= 1.0 / len(batch)

                # add regularization gradient
                if self.l1_reg != 0.0 or self.l2_reg != 0.0:
                    reg_grad = self.reg_gradient(self.params, grad, self.approx_reg)
                    grad += reg_grad

                # record gradient norm, before gradient gets modified by various algorithms
                self.gnorm = grad.norm2()

                delta = grad

                # check if Adagrad controller is begin used
                adagrad = next((controller for controller in self.controllers if isinstance(controller, AdaGrad)), None)
                if adagrad is None:
                    delta *= self.step_size
                    self.delta = delta
                else:
                    # this controller will modify self.delta
                    self.delta = delta
                    adagrad.control(self)

                # these controllers will modify self.delta, and maybe also self.halt
                for controller in self.controllers:
                    if isinstance(controller, AdaGrad):
                        continue
                    controller.control(self)

                # update params
                self.params += self.delta

                # check if unit normalization controller
                unit_norm = next((controller for controller in self.controllers if isinstance(controller, UnitNorm)), None)
                if unit_norm is not None:
                    unit_norm.control(self)

                self.track()

                self.steps += 1

                if self.halt:
                    return self.params

    def reg_gradient(self, params, grad, approx_reg):

        if approx_reg:
            # update just observed features
            features = grad.iterkeys()
        else:
            # update all features
            features = params.iterkeys()

        reg_grad = SparseVector()
        for feature in features:
            # might be in grad, but not in params
            if feature not in params:
                continue

            rgrad = -self.l1_reg * np.sign(params[feature])
            rgrad += -self.l2_reg * 2.0 * params[feature]
            reg_grad[feature] = rgrad

        return reg_grad


def gradient_check(f, f_grad_x, x, direction=None, verbose=False, precision=1e-4):
    """
    Check that finite difference estimates of slope converge to <f_grad_x, unit_vec> as ||delta|| -> 0
    f_grad_x is a sparse vector, representing the gradient of f evaluated at x
    """

    if direction is None:
        # initialize random direction
        direction = SparseVector.random(x)

    # normalize to be unit vector
    delta = direction * (1.0 / direction.norm2())

    # compute slope in direction of delta
    slope = f_grad_x.dot(delta)

    for k in range(20):
        slope_hat = (f(x + delta) - f(x)) / delta.norm2()
        diff = abs(slope - slope_hat)

        if verbose:
            print '|{} - {}| = {}'.format(slope, slope_hat, diff)

        # the diff must be smaller than some percentage of the theoretical slope
        if diff <= abs(slope) * precision:
            return True

        # keep halving the length of delta
        delta *= 0.5

    return False


def gradient_check_partials(f, f_grad_x, x, **kwargs):
    # test partial gradients, one for each parameter
    all_passed = True

    for key, val in x.iteritems():
        if isinstance(val, float):
            rand_val = np.random.normal()
        else:
            rand_val = np.random.normal(size=val.shape)

        direction = SparseVector({key: rand_val})
        result = gradient_check(f, f_grad_x, x, direction=direction, **kwargs)

        print key
        print result

        all_passed = all_passed and result

    return all_passed


def gradient_check_online(online_fxn, params, ex, partials=False, **kwargs):
    f = lambda p: online_fxn.value(p, ex)
    f_grad = online_fxn.gradient(params, ex)

    if partials:
        return gradient_check_partials(f, f_grad, params, **kwargs)
    else:
        return gradient_check(f, f_grad, params, **kwargs)


class Controller(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def control(self, maximizer):
        return


class Observer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def observe(self, maximizer):
        return


class BasicController(Controller):

    def __init__(self, report_wait=30, save_wait=30, max_steps=30000):
        self.report_wait = report_wait
        self.save_wait = save_wait
        self.max_steps = max_steps

    def control(self, maximizer):
        if maximizer.steps >= self.max_steps:
            print 'Halted after reaching max steps.'
            maximizer.halt = True

        if maximizer.steps % self.report_wait == 0:
            epochs = float(maximizer.steps * maximizer.batch_size) / len(maximizer.train)
            print 'steps: {}, epochs: {:.2f}'.format(maximizer.steps, epochs)
            util.metadata('steps', maximizer.steps)
            util.metadata('epochs', epochs)

            # report last seen
            time_rep = datetime.now().strftime('%H:%M:%S %m/%d')
            util.metadata('last_seen', time_rep)

            # report memory used
            util.metadata('gb_used', util.gb_used())

        if maximizer.steps % self.save_wait == 0 and maximizer.steps != 0:
            print 'saving params...'
            with open('params.cpkl', 'w') as f:
                # convert params to picklable format
                params = SparseVector(maximizer.params.as_dict())
                pickle.dump(params, f)


class AdaGrad(Controller):
    def __init__(self):
        self.step_cache = SparseVector()

    # Warning: This function assumes that maximizer.delta contains the gradient
    # of the function!
    @util.profile
    def control(self, maximizer):
        for param in maximizer.delta:
            if param in self.step_cache:
                self.step_cache[param] += maximizer.delta[param] ** 2
            else:
                self.step_cache[param] = maximizer.delta[param] ** 2

            maximizer.delta[param] = maximizer.step_size * maximizer.delta[param] / (np.sqrt(self.step_cache[param]) + 1e-8)


class UnitNorm(Controller):
    @util.profile
    def control(self, maximizer):
        # only update the entity parameters that received a gradient update
        for param in maximizer.delta:
            if isinstance(param, tuple):
                if param[0] == 'e':
                    maximizer.params[param] /= np.linalg.norm(maximizer.params[param])


class DeltaClipper(Controller):

    def __init__(self):
        self.delta_norms = []

    def control(self, maximizer):
        dnorm = maximizer.delta.norm2()
        self.delta_norms.append(dnorm)

        # get median so far
        self.delta_norms.sort()
        median = self.delta_norms[len(self.delta_norms) / 2]

        # if thresh exceeded, make a median-sized update
        thresh = 3.0 * median
        if dnorm > thresh:
            maximizer.delta *= median / dnorm

            # PER ENTRY VERSION
            # delta = maximizer.delta
            # for key, val in delta.iteritems():
            #     delta[key] = np.minimum(val, 1.0)


class NormObserver(Observer):
    def __init__(self, report_wait=30):
        self.report_wait = report_wait

    def observe(self, maximizer):
        if maximizer.steps % self.report_wait != 0:
            return None
        pnorm = maximizer.params.norm2()
        dnorm = maximizer.delta.norm2()
        gnorm = maximizer.gnorm
        return {('params_norm', 'params'): pnorm, ('norm', 'delta'): dnorm, ('norm', 'grad'): gnorm}


class SpeedObserver(Observer):
    def __init__(self, report_wait=30):
        self.report_wait = report_wait
        self.prev_steps = 0
        self.prev_time = time.time()

    def observe(self, maximizer):
        if maximizer.steps % self.report_wait != 0:
            return None
        seconds = time.time() - self.prev_time
        steps = maximizer.steps - self.prev_steps
        self.prev_time = time.time()
        self.prev_steps = maximizer.steps
        return {('speed', 'speed'): steps / seconds}


class ObjectiveObserver(Observer):
    def __init__(self, dset_samples, report_wait):
        self.dset_samples = dset_samples
        self.report_wait = report_wait

    def observe(self, maximizer):
        if maximizer.steps % self.report_wait == 0:
            def objective_mean(dset):
                sample = util.sample_if_large(dset, self.dset_samples)
                vals = [maximizer.objective.value(maximizer.params, ex) for ex in util.verboserate(sample)]
                return np.mean(vals)

            # Note that we never report exact on train
            return {('objective', 'train'): objective_mean(maximizer.train),
                    ('objective', 'dev'): objective_mean(maximizer.dev)}
        return None


class GradientChecker(Observer):
    def __init__(self, name, report_wait):
        self.name = name
        self.report_wait = report_wait

    def observe(self, maximizer):
        if maximizer.steps % self.report_wait == 0:
            print 'checking gradient'
            result = gradient_check_online(maximizer.objective, maximizer.params, maximizer.train[0], verbose=True)
            print result
        return None

