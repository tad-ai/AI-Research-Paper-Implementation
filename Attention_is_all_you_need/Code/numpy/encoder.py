import numpy as np
from multihead_attention import MultiHeadAttention
from layer_cal import layer_norm,feed_forward

def encoder_layer(attention_input,d_model,dff):
    attention_out=MultiHeadAttention(num_heads=8,d_model=d_model,embedding_vector=attention_input,mask=None)

    norm_input1=attention_out+attention_input
    gamma,beta=np.ones(d_model),np.zeros(d_model)
    norm1=layer_norm(norm_input1,gamma,beta)

    encoder_out=feed_forward(norm1,dff,d_model)

    norm_input2=norm1+encoder_out
    norm2=layer_norm(norm_input2,gamma,beta)  

    return norm2

def encoder(input,num_layers,d_model,dff):
    attntion_in=input
    for _ in range(num_layers):
        attntion_in=encoder_layer(attntion_in,d_model,dff)

    return attntion_in