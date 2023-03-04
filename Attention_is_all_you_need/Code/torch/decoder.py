import torch.nn as nn
from attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_head,dff,dropout):
        super(DecoderLayer,self).__init__()

        self.head_dim=d_model//n_head #d_k,d_v

        self.dec_masked_attn=MultiHeadAttention(d_model,n_head,self.head_dim,self.head_dim)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dec_attn=MultiHeadAttention(d_model,n_head,self.head_dim,self.head_dim)
        self.norm3=nn.LayerNorm(d_model)

        self.feed_forward=nn.Sequential(
            nn.Linear(d_model,dff*d_model),
            nn.ReLU(),
            nn.Linear(dff*d_model,d_model)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,dec_input,enc_output):
        masked_att_out=self.dec_masked_attn(dec_input,dec_input,dec_input)
        norm1=self.dropout(self.norm1(dec_input+masked_att_out))

        multi_att_out=self.dec_attn(norm1,enc_output,enc_output)
        norm2=self.dropout(self.norm2(norm1+multi_att_out))

        forward_net=self.feed_forward(norm2)
        norm3=self.dropout(self.norm3(forward_net+norm2))

        return norm3
    
class Decoder(nn.Module):
    def __init__(
            self,
            n_layers,
            d_model,
            n_head,
            dff,
            dropout
            ):
        super(Decoder,self).__init__()
        self.layers=nn.ModuleList(
            [
            DecoderLayer(
                d_model,
                n_head,
                dff,
                dropout
            ) 
            for _ in range(n_layers)]
        )

    def forward(self,dec_input1,enc_ouput1):
        dec_final=dec_input1

        for layer in self.layers:
            dec_final=layer(dec_input1,enc_ouput1)

        return dec_final
       
    
