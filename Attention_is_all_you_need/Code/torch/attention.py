import numpy as np
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_k,d_v):
        super(MultiHeadAttention,self).__init__()
        self.n_heads=n_heads
        self.d_k=d_k
        self.d_v=d_v
        self.W_Q=nn.Linear(d_model,d_k*n_heads)
        self.W_K=nn.Linear(d_model,d_k*n_heads)
        self.W_V=nn.Linear(d_model,d_v*n_heads)

    def forward(self,Q,K,V,attn_mask):
        residual,batch_size=Q,Q.size(0)
        q_s=self.W_Q(Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        k_s=self.W_K(K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        v_s=self.W_V(V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super(ScaledDotProduct).__init__()


    def forward(self,Q,K,V):
        d_k=K.size()[-1]

        att_score=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        att_score=nn.softmax(dim=-1)(att_score)
        att_score=torch.matmul(att_score,V)

        return att_score
