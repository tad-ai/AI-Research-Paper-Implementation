import numpy as np
import torch

def PositionalEncoding(max_position_embeddings,d_model):
    sinusoid_table = np.array(
            [
                [p / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
                for p in range(max_position_embeddings)
            ]
        )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) #(max_position_embeddings, d_model)
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) #odd indexes in the emb dimension = cos

    return torch.FloatTencos(sinusoid_table)