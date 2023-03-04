import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size,d_model,src_len,trg_len,n_head,dff,dropout):
        super(Transformer).__init__()
        self.src_emb=nn.Embedding(src_vocab_size,d_model)
        self.src_pos_emb=nn.Embedding.from_pretrained(PositionalEncoding(src_len,d_model),freeze=True)

        self.trg_emb=nn.Embedding(trg_vocab_size,d_model)
        self.trg_pos_emb=nn.Embedding.from_pretrained(PositionalEncoding(trg_len,d_model),freeze=True)

        self.encoder=Encoder(d_model=d_model,n_head=n_head,dff=dff,dropout=dropout)
        self.decoder=Decoder(d_model=d_model,n_head=n_head,dff=dff,dropout=dropout)
        self.projection=nn.Linear(d_model,trg_len,bias=False)

    def forward(self,enc_input,dec_input):
        enc_output=self.src_emb(enc_input)+self.src_pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        enc_output=self.encoder(enc_output)

        dec_output=self.trg_emb(dec_input)+self.src_pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        dec_output=self.decoder(dec_output,enc_output)

        dec_logits=self.projection(dec_output)
        tran_output=dec_logits.view(-1, dec_logits.size(-1))

        return tran_output