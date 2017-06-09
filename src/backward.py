import copy
import numpy
from sklearn.metrics import mean_squared_error


class BackwardSelector(object):
    def __init__(self, model, n_features):

        self.model = model
        self.n_features = n_features

    def fit(self, Xtrain, ytrain, Xval, yval):
        # different from the forward feature selection - features starts as an
        # array of ones (as I will then blacklist columns 1 by 1 - making each a 0)
        features = numpy.ones(Xtrain.shape[1], dtype=numpy.int)
        valerrors = []

        # incrementally blacklist features until only n_features remain
        for i in xrange(Xtrain.shape[1] - self.n_features):
            print("Blacklisting a new feature ...")

            # compute validation errors for new potential 
            # features and return worst feature index and error
            worst_idx, worst_error = self._compute_worst_feature(Xtrain, ytrain, Xval, yval, features)
            print("Worst feature index is: %i. Worst validation error is: %f" % (worst_idx, worst_error))

            # select worst feature
            features[worst_idx] = 0

            # store validation error (for plotting purposes)
            valerrors.append(worst_error)

        # update features and validation errors
        self.features = features
        self.valerrors = valerrors

        # recompute final model 
        self.model.fit(Xtrain[:, self.features.astype(bool)], ytrain)

    def predict(self, X):

        return self.model.predict(X[:, self.features.astype(bool)])

    def get_validation_errors(self):

        return self.valerrors

    def _compute_worst_feature(self, Xtrain, ytrain, Xval, yval, features):

        valerrors = numpy.inf * numpy.ones(Xtrain.shape[1], dtype=numpy.float)

        for j in xrange(Xtrain.shape[1]):

            # only check/add feature in case it is not selected yet
            if features[j] != 0:
                current_features = copy.deepcopy(features)
                current_features[j] = 0
                # fit model and compute validation error
                self.model.fit(Xtrain[:, current_features.astype(bool)], ytrain)
                preds = self.model.predict(Xval[:, current_features.astype(bool)])
                err = mean_squared_error(yval, preds)
                print("\tValidation error without feature %i: %f" % (j, err))
                valerrors[j] = err

        # compute index with smallest validation error
        worst_idx = numpy.argmin(valerrors)
        worst_err = numpy.min(valerrors)

        return worst_idx, worst_err
