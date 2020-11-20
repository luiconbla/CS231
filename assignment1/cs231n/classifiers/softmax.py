from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_trains = X.shape[0]

    for n in range(num_trains):
      scores = X[n,:].dot(W)
      scores -= np.amax(scores)
      exp_scores = np.exp(scores)
      confidences = exp_scores / np.sum(exp_scores)
      loss -= np.log( confidences[y[n]] )
      dW[:,y[n]] -= X[n]
      dW += np.outer(X[n], confidences)
    loss /= num_trains
    loss += reg * np.sum(W**2)
    dW /= num_trains
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_trains = X.shape[0]
    num_classes = W.shape[1]
    arange_trains = np.arange(num_trains)

    scores = X.dot(W)
    scores -= np.amax(scores, axis=1)[:,None]
    exp_scores = np.exp(scores)
    confidences = exp_scores / np.sum(exp_scores,axis=1)[:,None]
    
    loss = -np.sum( np.log( confidences[arange_trains,y] ) ) / num_trains + reg * np.sum(W**2)
    
    true_classes = np.zeros((num_trains,num_classes))
    true_classes[arange_trains,y] = 1
    
    dW = X.T.dot(confidences-true_classes) / num_trains + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
