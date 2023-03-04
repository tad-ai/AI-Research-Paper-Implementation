import numpy as np
from positional_encoding import PositionalEncoding
from multihead_attention import softmax
from encoder import encoder
from decoder import decoder

#Crete a dataset
sentences = [['quiero esto P P', 'S i want this', 'i want this E'],['ella es hermosa P','S she is beautiful','she is beautiful E']]


src_vocab = {'P': 0,'ella':1, 'esto': 1, 'quiero': 2, 'hermosa':3,'es':4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1,'she':2, 'want': 3,'beautiful':4, 'is':5, 'this': 6,'S':7, 'E': 8}
tgt_vocab_size = len(tgt_vocab)
number_dict = {i: w for i, w in enumerate(tgt_vocab)}

d_model=512 #Embedding size
n_layers=6 #Number of encoder or decoder layers
batch_size=2 #Batch size
sequnce_length=4 #Maximum length of a sequence
d_ff=2048 #Feedforward dimension


def make_batch(sentences):
    input_batch=[]
    output_batch=[]
    target_batch=[]
    for i in range (len(sentences)):
        input_batch.append([src_vocab[n] for n in sentences[i][0].split()])
        output_batch.append([tgt_vocab[n] for n in sentences[i][1].split()])
        target_batch.append([tgt_vocab[n] for n in sentences[i][2].split()])
    return input_batch,output_batch,target_batch

enc_inputs, dec_inputs, target_final = make_batch(sentences)

def one_hot_vector(num_words,enc_inputs):
    one_hot=[]
    for j in range (len(enc_inputs)):
        one_hot_list = np.zeros((len(enc_inputs[0]), num_words))

        # Convert the indices into one-hot vectors
        for i,index in enumerate(enc_inputs[j]):
            one_hot_list[i, index] = 1
        
        one_hot.append(one_hot_list)
    
    return one_hot

enc_one_hot_vector=one_hot_vector(src_vocab_size,enc_inputs)
enc_one_hot_vector=np.array(enc_one_hot_vector)
weights = np.random.rand(len(sentences),src_vocab_size, d_model)
word_embeddings=np.matmul(enc_one_hot_vector,weights)
position_embeddings=PositionalEncoding(4,d_model=d_model)
embeddings = word_embeddings + position_embeddings

tr_enc_one_hot_vector=one_hot_vector(tgt_vocab_size,dec_inputs)
tr_enc_one_hot_vector=np.array(tr_enc_one_hot_vector)
tr_weights = np.random.rand(len(sentences),tgt_vocab_size, d_model)
tr_word_embeddings=np.matmul(tr_enc_one_hot_vector,tr_weights)
tr_position_embeddings=PositionalEncoding(4,d_model=d_model)
tr_embeddings = tr_word_embeddings + tr_position_embeddings

def transformer():
    encoder_out=encoder(embeddings,n_layers,512,d_ff)

    decoder_out=decoder(tr_embeddings,encoder_out,n_layers,512,d_ff)

    W_linear=np.random.rand(2,d_model,tgt_vocab_size)
    linear=np.matmul(decoder_out,W_linear)

    final_out=softmax(linear)

transformer()
