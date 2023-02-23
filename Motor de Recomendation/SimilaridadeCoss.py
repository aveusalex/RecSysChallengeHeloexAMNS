# calculando a similaridade cosseno entre os embeddings

import numpy as np
import pandas as pd


def similaridade_cosseno(embedding, embedding2):
    # calcula a similaridade cosseno entre dois vetores
    return np.dot(embedding, embedding2) / (np.linalg.norm(embedding) * np.linalg.norm(embedding2))

