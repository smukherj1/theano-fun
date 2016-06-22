from LogisticRegression import _LogisticRegressionModel
import theano
from theano import tensor as T
import numpy as np
import math

class _HiddenLayerModel(object):
    def __init__(self, d_input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.d_input = d_input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(self.d_input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class _MLPModel(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, d_input, n_in, n_hidden, n_out, activation=T.tanh):
        """Initialize the parameters for the multilayer perceptron

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        num_hidden_layers = 1
        #hidden_layer_sizes = (n_hidden)
        self.hiddenLayers = [_HiddenLayerModel(
            d_input=d_input,
            n_in=n_in,
            n_out=n_hidden,
            activation=activation
        )]
        for i in range(1, num_hidden_layers):
            self.hiddenLayers.append(
                _HiddenLayerModel(
                    d_input=self.hiddenLayers[i - 1].output,
                    n_in=n_hidden,
                    n_out=n_hidden,
                    activation=activation
                )
            )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = _LogisticRegressionModel(
            d_input=self.hiddenLayers[-1].output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()

        for ihidden_layer in self.hiddenLayers:
            self.L1 += abs(ihidden_layer.W).sum()
            self.L2_sqr += (ihidden_layer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        # same holds for the function computing the number of errors
        #self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = []
        for ihidden_layer in self.hiddenLayers:
            self.params += ihidden_layer.params
        self.params += self.logRegressionLayer.params
        # end-snippet-3

        self.y_pred = self.logRegressionLayer.y_pred

        # keep track of model input
        self.d_input = d_input

    def negative_log_likelihood(self, y):
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        return self.logRegressionLayer.negative_log_likelihood(y)


class MLP(object):
    def __init__(self, n_in, n_hidden, n_out, num_hidden_layers=1, beta=0.01, 
        L1_reg=0.00, L2_reg=0.0001, activation=T.tanh):
        
        if n_hidden == None:
            n_hidden = max(n_in, n_out)

        self.__beta = beta
        self.__n_in = n_in
        self.__n_hidden = n_hidden
        self.__n_out = n_out
        self.__num_hidden_layers = num_hidden_layers

        self.__x = T.dmatrix('x')
        self.__y = T.dmatrix('y')

        self.__mlp_model = _MLPModel(d_input=self.__x, 
            n_in=self.__n_in,
            n_hidden=self.__n_hidden,
            n_out=self.__n_out,
            activation=activation)
        self.__cost = (self.__mlp_model.negative_log_likelihood(self.__y) \
            + L1_reg * self.__mlp_model.L1 \
            + L2_reg * self.__mlp_model.L2_sqr
        )

        self.__gparams = [T.grad(self.__cost, param) for param in self.__mlp_model.params]
        updates = [
            (param, param - self.__beta * gparam)
            for param, gparam in zip(self.__mlp_model.params, self.__gparams)
        ]

        self.__train_model = theano.function(
            inputs=[self.__x, self.__y],
            outputs=[self.__cost, self.__mlp_model.y_pred],
            updates=updates
        )

        self.__prediction_model = theano.function(
            inputs=[self.__x],
            outputs=self.__mlp_model.y_pred
        )
        return

    def fit(self, x, y):
        cost, pred = self.__train_model(x, y)
        if math.isnan(cost):
            raise RuntimeError('Cost is nan')
        return cost

    def predict(self, x):
        prob_dist = self.__prediction_model(x)
        result = np.zeros(prob_dist.shape)
        for ilabel, label in enumerate(prob_dist):
            result[ilabel][np.argmax(label)] = 1.
        return result