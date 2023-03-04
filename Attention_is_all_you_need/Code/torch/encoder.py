import torch.nn as nn
from attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_head,dff,dropout):
        super(EncoderLayer).__init__()
        self.head_dim=d_model//n_head #d_k,d_v

        self.enc_attn=MultiHeadAttention(d_model,n_head,self.head_dim,self.head_dim)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

        self.feed_forward=nn.Sequential(
            nn.Linear(d_model,dff*d_model),
            nn.ReLU(),
            nn.Linear(dff*d_model,d_model)
        )
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,enc_input):
        attention_out=self.enc_attn(enc_input,enc_input,enc_input)
        norm_1=self.dropout(self.norm1(enc_input+attention_out))

        forward_net=self.feed_forward(norm_1)

        norm_2=self.dropout(self.norm2(forward_net+norm_1))
        
        return norm_2


class Encoder(nn.Module):
    def __init__(
            self,
            n_layers,
            d_model,
            n_head,
            dff,
            dropout
            ):
        super(Encoder,self).__init__()
        self.layers=nn.ModuleList(
            [
            EncoderLayer(
                d_model,
                n_head,
                dff,
                dropout
            ) 
            for _ in range(n_layers)]
        )

    def forward(self,enc_input1):
        enc_output=enc_input1

        for layer in self.layers:
            enc_output=layer(enc_output)

        return enc_output
    
