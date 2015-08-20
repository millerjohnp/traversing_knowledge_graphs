import warnings

import numpy as np


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import theano

from optimize import OnlineFunction, SparseVector
import util
import models
from data import load_glove_vectors


class CompositionalModel(OnlineFunction):
    def __init__(self, negative_generator, path_model='bilinear', objective='margin',
                 word_vectors=False):
        self.objective = objective
        self.path_model = path_model
        self.word_vectors = word_vectors

        # used to sample negative examples. Note: must be deterministic
        # for gradient checking to pass!
        self.negative_generator = negative_generator

        print 'compiling path traversal functions'
        theano.config.allow_gc = False
        path_functions = models.get_path_query_model(path_model, objective)
        self.score = path_functions['score']
        self.fprop = path_functions['fprop']
        self.bprop = path_functions['bprop']
        print 'done'

    @util.profile
    def value(self, params, ex):
        value = 0.
        x_s, W, x_t = self.unpack_pathQuery(params, ex)

        # sample negative target examples
        samples = self.negative_generator(ex, 't')
        if len(samples) > 0:
            negatives = self.unpack_entities(params, samples)
            batch_X_t = np.append(x_t, negatives, axis=1)

            if self.path_model is 'NTN':
                u, V, b = self.get_ntn_params(params, ex)
                value += self.fprop(x_s, W, batch_X_t, u, V, b)
            else:
                value += self.fprop(x_s, W, batch_X_t)

        # sample negative source examples
        if self.path_model is not 'NTN':
            samples = self.negative_generator(ex, 's')
            if len(samples) > 0:
                negatives = self.unpack_entities(params, samples)
                batch_X_s = np.append(x_s, negatives, axis=1)

                if self.path_model is 'transE':
                    # transE expects the source to be a column vector
                    # Flip and negate the source/targets to ensure the
                    # dimensions match without changing the math
                    value += self.fprop(-x_t, W, -batch_X_s)
                else:
                    value += self.fprop(batch_X_s, W, x_t)

        return value

    @util.profile
    def gradient(self, params, ex):
        grad = SparseVector()
        x_s, W, x_t = self.unpack_pathQuery(params, ex)

        # sample negative target examples
        samples = self.negative_generator(ex, 't')
        if len(samples) > 0:
            negatives = self.unpack_entities(params, samples)
            batch_X_t = np.append(x_t, negatives, axis=1)

            if self.path_model is 'NTN':
                u, V, b = self.get_ntn_params(params, ex)
                gx_s, gW, gBatch_X_t, gu, gV, gb = self.bprop(x_s, W, batch_X_t, u, V, b)
                self.update_ntn_params(grad, ex, gu, gV, gb)
            else:
                gx_s, gW, gBatch_X_t = self.bprop(x_s, W, batch_X_t)

            self.accumulate_entity_gradients(grad, ex.s, gx_s)
            self.accumulate_relation_gradients(grad, ex, gW)
            self.accumulate_entity_gradients(grad, [ex.t] + samples, gBatch_X_t)

        # sample negative subject examples
        if self.path_model is not 'NTN':
            samples = self.negative_generator(ex, 's')
            if len(samples) > 0:
                negatives = self.unpack_entities(params, samples)
                batch_X_s = np.append(x_s, negatives, axis=1)

                if self.path_model is 'transE':
                    # transE expects the source to be a column vector
                    # Flip and negate the source/targets to ensure the
                    # dimensions match without changing the math
                    gx_t, gW, gBatch_X_s = self.bprop(-x_t, W, -batch_X_s)
                    gx_t, gBatch_X_s = -gx_t, -gBatch_X_s
                else:
                    gBatch_X_s, gW, gx_t = self.bprop(batch_X_s, W, x_t)

                self.accumulate_entity_gradients(grad, [ex.s] + samples, gBatch_X_s)
                self.accumulate_relation_gradients(grad, ex, gW)
                self.accumulate_entity_gradients(grad, ex.t, gx_t)

        if self.word_vectors:
            # backprop the entity vector gradients to the constituent word vecs
            grad = self.get_word_vector_gradients(grad, params, ex)

        return grad

    def predict(self, params, ex):
        x_s, W, x_t = self.unpack_pathQuery(params, ex)
        if self.path_model is 'NTN':
            u, V, b = self.get_ntn_params(params, ex)
            return self.score(x_s, W, x_t, u, V, b)
        else:
            return self.score(x_s, W, x_t)

    def accumulate_entity_gradients(self, grad, entities, gE):
        if isinstance(entities, list):
            for idx, ent in enumerate(entities):
                update = gE[:, idx].reshape(-1, 1)
                self.add_to_gradient(grad, ('e', ent), update)
        else:
            self.add_to_gradient(grad, ('e', entities), gE)

    def accumulate_relation_gradients(self, grad, ex, gW):
        if self.path_model is 'NTN':
                self.add_to_gradient(grad, ('r', ex.r[0]), gW)
        elif self.path_model in ['bilinear_diag', 'transE']:
            for idx, relation in enumerate(ex.r):
                update = gW[:, idx].reshape(-1, 1)
                self.add_to_gradient(grad, ('r', relation), update)
        elif self.path_model is 'bilinear':
            for idx, relation in enumerate(ex.r):
                self.add_to_gradient(grad, ('r', relation), gW[idx])
        else:
            raise NotImplementedError()

    def update_ntn_params(self, grad, ex, gu, gV, gb):
        r = ex.r[0]
        self.add_to_gradient(grad, ('NTN', 'u', r), gu)
        self.add_to_gradient(grad, ('NTN', 'V', r), gV)
        self.add_to_gradient(grad, ('NTN', 'b', r), gb)

    def add_to_gradient(self, grad, key, value):
        if key in grad:
            grad[key] += value
        else:
            grad[key] = value

    def get_word_vector_gradients(self, entity_vec_grad, params, ex):
        '''
            Replaces gradients on entity vectors with the corresponding
            gradients on individual word vectors
        '''
        new_grad = SparseVector()
        for (ftype, ent), g in entity_vec_grad.iteritems():
            if ftype != 'e':
                new_grad[(ftype, ent)] = g
            else:
                words = self.get_entity_words(ent)
                for word in words:
                    if ('wv', word) in new_grad:
                        new_grad[('wv', word)] += g / len(words)
                    else:
                        new_grad[('wv', word)] = g / len(words)

        return new_grad

    def unpack_pathQuery(self, params, ex):
        X_s = self.unpack_entities(params, ex.s)
        W = self.unpack_relations(params, ex.r)
        X_t = self.unpack_entities(params, ex.t)
        return X_s, W, X_t

    def unpack_entities(self, params, entities):
        if not isinstance(entities, list):
            return self.build_entity_vector(params, entities)
        elif len(entities) > 1:
            entities = [self.build_entity_vector(params, entity) for entity in entities]
            return np.asarray(entities).squeeze().T
        else:
            return self.build_entity_vector(params, entities[0])

    def unpack_relations(self, params, r_tuple):
        if self.path_model is 'NTN':
            W = params[('r', r_tuple[0])]
        elif self.path_model in ['bilinear_diag', 'transE']:
            W_list = [params[('r', rel)] for rel in r_tuple]
            W = np.concatenate((W_list), axis=1)
        elif self.path_model is 'bilinear':
            # returns a tensor of size len(W_list) x dim x dim
            W_list = [params[('r', rel)] for rel in r_tuple]
            W = np.rollaxis(np.dstack(W_list), -1)
        else:
            raise NotImplementedError()

        return W

    def get_ntn_params(self, params, ex):
        ''' Return u, V, b'''
        r = ex.r[0]
        return params[('NTN', 'u', r)], params[('NTN', 'V', r)], params[('NTN', 'b', r)]

    @staticmethod
    def get_entity_words(entity):
        if entity[:2] == '__':
            # wordnet example, so remove the "__" at the start
            entity = entity[2:]

            #remove the word sense
            entity = entity[:entity.rfind('_')]

        return entity.split('_')

    def build_entity_vector(self, params, entity):
        if not self.word_vectors:
            # ignore the word vectors and just return the plain entity
            return params[('e', entity)]

        words = self.get_entity_words(entity)
        vector = 0.
        for word in words:
            vector += params[('wv', word)]
        return vector / len(words)

    @staticmethod
    def init_params(entity_list, relations_list, embed_dim, model='bilinear',
                    hidden_dim=20, init_scale=1.0, glove_path=None):
        '''
            If GLOVE_PATH is None, assumes entity vector representation.
            Otherwise, uses the word vectors found at GLOVE_PATH
        '''
        params = SparseVector()

        if glove_path is not None:
            print "Initializing word vectors from Glove..."
            wv_vecs = load_glove_vectors(glove_path)

        for entity in entity_list:
            if glove_path is not None:
                # use word vector representation
                entity_words = CompositionalModel.get_entity_words(entity)
                for word in entity_words:
                    if word in wv_vecs:
                        params[('wv', word)] = wv_vecs[word]
                    else:
                        params[('wv', word)] = init_scale * np.random.randn(embed_dim, 1)
            else:
                params[('e', entity)] = init_scale * np.random.randn(embed_dim, 1)

        for relation in relations_list:
            if model is 'bilinear':
                params[('r', relation)] = init_scale * np.random.randn(embed_dim, embed_dim)
            elif model in ['bilinear_diag', 'transE']:
                params[('r', relation)] = init_scale * np.random.randn(embed_dim, 1)
            elif model is 'NTN':
                # using initialization scale from Socher NTN
                r = 0.001
                params[('r', relation)] = np.random.rand(hidden_dim, embed_dim, embed_dim) * 2 * r - r
                params[('NTN', 'V', relation)] = np.random.rand(hidden_dim, 2 * embed_dim) * 2 * r - r
                params[('NTN', 'b', relation)] = np.random.rand(hidden_dim, 1) * 2 * r - r
                params[('NTN', 'u', relation)] = np.random.rand(hidden_dim, 1) * 2 * r - r
            else:
                raise NotImplementedError()

        return params
