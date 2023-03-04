import torch
from transformer import Transformer
import torch.nn as nn
import torch.optim as optim

sentences = [['quiero esto P P', 'P i want this', 'i want this E'],['ella es hermosa P','P she is beautiful','she is beautiful E']]

src_vocab = {'P': 0,'ella':1, 'esto': 1, 'quiero': 2, 'hermosa':3,'es':4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1,'she':2, 'want': 3,'beautiful':4, 'is':5, 'this': 6, 'E': 9}
tgt_vocab_size = len(tgt_vocab)
number_dict = {i: w for i, w in enumerate(tgt_vocab)}


def make_batch(sentences):
    input_batch=[]
    output_batch=[]
    target_batch=[]
    for i in range (len(sentences)):
        input_batch.append([src_vocab[n] for n in sentences[i][0].split()])
        output_batch.append([tgt_vocab[n] for n in sentences[i][1].split()])
        target_batch.append([tgt_vocab[n] for n in sentences[i][2].split()])
    return torch.LongTensor(input_batch),torch.LongTensor(output_batch),torch.LongTensor(target_batch)

enc_inputs, dec_inputs, target_final = make_batch(sentences)

d_model=512
src_len=4
trg_len=4
n_head=8
dff=2048
dropout=0.1

model=Transformer(src_vocab_size,tgt_vocab_size,d_model,src_len,trg_len,n_head,dff,dropout)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    final_output=model(enc_inputs,dec_inputs)
    loss=criterion(final_output)
    loss.backward()
    optimizer.step()

# Test
predict, _, _, _ = model(enc_inputs, dec_inputs)
predict = predict.data.max(1, keepdim=True)[1]