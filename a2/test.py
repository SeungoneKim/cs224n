def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s= 1/(1+np.exp(x))
    ### END YOUR CODE

    return s
import numpy as np
a= np.random.rand(4,1)
print(a)
print()
print(sigmoid(a))