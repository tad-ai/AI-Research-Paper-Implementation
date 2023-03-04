import numpy as np
import torch.nn as nn
import math

def softmax(Z):
    ex=np.exp(Z-np.max(Z))
    return ex/ex.sum()

def MultiHeadAttention(num_heads,d_model,embedding_vector,mask=None,deco_v=[]):
    head_dim=d_model//num_heads

    W_Q=np.random.rand(2,d_model,d_model)
    W_V=np.random.rand(2,d_model,d_model)
    W_K=np.random.rand(2,d_model,d_model)
    W_H=np.random.rand(2,d_model,d_model)

    Q=np.matmul(embedding_vector,W_Q)

    if deco_v==[]:
        V=np.matmul(embedding_vector,W_V)
        K=np.matmul(embedding_vector,W_K)
    else:
        V=np.matmul(deco_v,W_V)
        K=np.matmul(deco_v,W_K)

    Q=np.array(np.split(Q, num_heads, axis=-1)).transpose(1, 0, 2, 3)
    V=np.array(np.split(V, num_heads, axis=-1)).transpose(1, 0, 2, 3)
    K=np.array(np.split(K, num_heads, axis=-1)).transpose(1, 0, 2, 3)

    d_k=len(K[-1])
    att_scores=np.matmul(Q,K.transpose(0,1,3,2))
    if mask is not None:
        att_scores[:,:,mask==0]= -math.inf
        print("inf att_scores:",str(att_scores))

    # att_scores = np.apply_along_axis(softmax, -1, att_scores)
    att_scores=att_scores/math.sqrt(d_k)
    att_scores=softmax(att_scores)
    att_scores=np.matmul(att_scores,V)

    head_final=att_scores.transpose(0,2,1,3).reshape(2,4,-1)
    head_final=np.matmul(head_final,W_H)
    return head_final

