import numpy as np


def layer_norm(input_tens, gamma, beta, epsilon=1e-8):
    '''
    gamma: Scaling parameter
    beta: Shift parameter
    '''

    mean = np.mean(input_tens, axis=-1, keepdims=True)
    variance = np.var(input_tens, axis=-1, keepdims=True)
    
    norm = (input_tens - mean) / np.sqrt(variance + epsilon)
    
    norm = gamma * norm + beta
    
    return norm

def ReLU(input_tens):
    return np.maximum(0,input_tens)

def feed_forward(input_tens,dff,d_model):
    h1=np.random.rand(d_model,dff)
    h2=np.random.rand(dff,d_model)

    output_tens=np.matmul(input_tens,h1)
    output_tens=ReLU(output_tens)
    output_tens=np.matmul(output_tens,h2)

    return output_tens