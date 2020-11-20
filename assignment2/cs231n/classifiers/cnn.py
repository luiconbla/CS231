from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # x => c1 => r1 => p1 => a2 => r2 => a3 => m3 => L/y_pred
        #
        # x  = N·C·H·W
        #
        # W1 = F·C·hc1·wc1
        # b1 = F
        # c1 = N·F·HC1·WC1  with HC1=(H+2*PC1-hc1)/SC1+1 and WC1=(W+2*PC1-wc1)/SC1+1
        #
        # r1 = N·F·HC1·WC1  (same)
        #
        # p1 = N·F·HP1·WP1  with HP1=(HC1-hp1)/SP1+1 and WP1=(WC1-wp1)/SP1+1
        #
        # W2 = (F·HP1·WP1)·(HD)
        # b2 = HD
        # a2 = N·HD
        #
        # r2 = N·HD  (same)
        #
        # W3 = HD·K
        # b3 = K
        # a3 = N·K
        # 
        # m3 = N·K  (same)

        C, H, W = input_dim
        
        # F = num_filters
        # hc1, lc1 = filter_size
        # HD = hidden_dim
        # K = num_classes
        # HC1, WC1 = H, W  so height & width of input is preserved
        # hp1, wp1 = 2 since 2x2 maxpool; I assume SP1=2
        assert H % 2 == 0, 'H=HC1 is not divisible by 2 from maxpool'
        assert W % 2 == 0, 'W=WC1 is not divisible by 2 from maxpool'
        HP1, WP1 = int(H/2), int(W/2)

        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)

        self.params['b1'] = np.zeros(num_filters)

        self.params['W2'] = weight_scale * np.random.randn(num_filters*HP1*WP1, hidden_dim)

        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)

        self.params['b3'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # x  = N·C·H·W
        #
        # W1 = F·C·hc1·wc1
        # b1 = F
        # c1 = N·F·HC1·WC1  with HC1=(H+2*PC1-hc1)/SC1+1 and WC1=(W+2*PC1-wc1)/SC1+1
        #
        # r1 = N·F·HC1·WC1  (same)
        #
        # p1 = N·F·HP1·WP1  with HP1=(HC1-hp1)/SP1+1 and WP1=(WC1-wp1)/SP1+1
        #
        # W2 = (F·HP1·WP1)·(HD)
        # b2 = HD
        # a2 = N·HD
        #
        # r2 = N·HD  (same)
        #
        # W3 = HD·K
        # b3 = K
        # a3 = N·K
        # 
        # m3 = N·K  (same)

        # x => c1->r1->p1 => a2->r2 => a3 => m3->L/y_pred

        crp1, crp1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        N, F, HP1, WP1 = crp1.shape
        crp1.resize((N, F, HP1, WP1))  # crp1 from N·F·HP1·WP1 to N·(F·HP1·WP1)

        ar2, ar2_cache = affine_relu_forward(crp1, W2, b2)

        a3, a3_cache = affine_forward(ar2, W3, b3)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return a3  # LCB changed from CS231 scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, d_a3 = softmax_loss(a3, y)
        loss += self.reg / 2 * ( np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2) )

        d_ar2, grads['W3'], grads['b3'] = affine_backward(d_a3, a3_cache)

        grads['W3'] += self.reg * W3

        d_crp1, grads['W2'], grads['b2'] = affine_relu_backward(d_ar2, ar2_cache)

        grads['W2'] += self.reg * W2
  
        d_crp1.resize((N, F, HP1, WP1))  # d_crp1 from N·(F·HP1·WP1) to N·F·HP1·WP1

        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(d_crp1, crp1_cache)

        grads['W1'] += self.reg * W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
