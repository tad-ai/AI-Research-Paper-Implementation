from multihead_attention import MultiHeadAttention
from layer_cal import layer_norm,feed_forward

import numpy as np

def decoder_layer(attention_input,v_enc,d_model,dff):
    attention_out=MultiHeadAttention(num_heads=8,d_model=d_model,embedding_vector=attention_input,mask="T")

    norm_input1=attention_out+attention_input
    gamma,beta=np.ones(d_model),np.zeros(d_model)
    norm1=layer_norm(norm_input1,gamma,beta)

    attention_out2=MultiHeadAttention(num_heads=8,d_model=d_model,embedding_vector=norm1,deco_v=v_enc)

    norm_input2=attention_out2+norm1
    norm2=layer_norm(norm_input2,gamma,beta)

    encoder_out=feed_forward(norm2,dff,d_model)

    norm_input3=norm2+encoder_out
    norm3=layer_norm(norm_input3,gamma,beta) 

    return norm3

def decoder(input,enc_out,num_layers,d_model,dff):
    attntion_in=input
    for _ in range(num_layers):
        attntion_in=decoder_layer(attntion_in,enc_out,d_model,dff)

    return attntion_in
