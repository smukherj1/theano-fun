import theano
import theano.tensor as T
import numpy
import math

class _LogisticRegressionModel(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, d_input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param d_input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.random.normal(
                size=(n_in, n_out)
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.raw_p_y_given_x = T.nnet.softmax(T.dot(d_input, self.W) + self.b)
        self.p_y_given_x = T.clip(self.raw_p_y_given_x, 1e-3, 0.95)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  representsute prediction as class whose
        # probability is maximal
        #self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_pred = self.p_y_given_x
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = d_input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        cross_entropy = T.nnet.binary_crossentropy(self.p_y_given_x, y)
        return T.mean(cross_entropy)
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class LogisticRegression(object):
    def __init__(self, beta=0.1, n_in=1, n_out=1):
        self.__beta = beta
        self.__x = T.dmatrix('x')
        self.__y = T.dmatrix('y')
        self.__n_in = n_in
        self.__n_out = n_out

        self.__clf_model = _LogisticRegressionModel(d_input=self.__x, 
            n_in=self.__n_in,
            n_out=self.__n_out)
        self.__cost = self.__clf_model.negative_log_likelihood(self.__y)

        # compute the gradient of cost with respect to theta = (W,b)
        self.__g_W = T.grad(cost=self.__cost, wrt=self.__clf_model.W)
        self.__g_b = T.grad(cost=self.__cost, wrt=self.__clf_model.b)

        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        self.__updates = [(self.__clf_model.W, 
            self.__clf_model.W - self.__beta * self.__g_W),
               (self.__clf_model.b, self.__clf_model.b 
                - self.__beta * self.__g_b)]

        self.__train_model = theano.function(
            inputs=[self.__x, self.__y],
            outputs=[self.__cost, self.__clf_model.y_pred, self.__g_W, self.__g_b],
            updates=self.__updates,
        )

        self.__prediction_model = theano.function(
            inputs=[self.__clf_model.input],
            outputs=self.__clf_model.y_pred
        )

    @staticmethod
    def __arrBad(arr):
        bad = numpy.isnan(arr).any() or numpy.isinf(arr).any()
        return bad

    def fit(self, x, y):
        #numpy.savetxt('W_before.txt', self.__clf_model.W.get_value())
        #numpy.savetxt('b_before.txt', self.__clf_model.b.get_value())
        cost, pred, g_W, g_b = self.__train_model(x, y)
        #numpy.savetxt('W_after.txt', self.__clf_model.W.get_value())
        #numpy.savetxt('b_after.txt', self.__clf_model.b.get_value())
        '''
        if math.isnan(cost) or self.__arrBad(pred) \
            or self.__arrBad(self.__clf_model.W.get_value()) \
            or self.__arrBad(self.__clf_model.b.get_value()):
            print cost
            numpy.savetxt('bad_g_W.txt', g_W)
            numpy.savetxt('bad_g_b.txt', g_b)
            numpy.savetxt('bad_pred.txt', pred)
            numpy.savetxt('bad_cost_arr.txt', cost_arr)
            exit()
        '''
        return cost

    def predict(self, x):
        prob_dist = self.__prediction_model(x)
        result = numpy.zeros(prob_dist.shape)
        for ilabel, label in enumerate(prob_dist):
            result[ilabel][numpy.argmax(label)] = 1.
        return result
