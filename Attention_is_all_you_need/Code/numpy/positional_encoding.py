import numpy as np

def PositionalEncoding(max_position_embeddings,d_model):
    weights_pe=np.random.random((max_position_embeddings, d_model))

    theta = np.array(
            [
                [p / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
                for p in range(max_position_embeddings)
            ]
        )
    weights_pe[:, 0::2] = np.sin(theta[:, 0::2]) #(max_position_embeddings, d_model)
    weights_pe[:, 1::2] = np.cos(theta[:, 1::2]) #odd indexes in the emb dimension = cos

    return weights_pe