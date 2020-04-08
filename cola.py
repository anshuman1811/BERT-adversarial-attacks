import numpy as np
import pandas as pd
import editdistance
import random
from unreadable_attack import shuffle_sentence, add_shuffle_sentence

path_to_cola = "glue_data/CoLA/dev_original.tsv"
cola = pd.read_csv(path_to_cola, sep="\t", header=None)

cola_shuff = cola.copy()
cola_add_shuff = cola.copy()

rand_degree = 0.4
shuffle_degree = 0.65
ind = np.arange(cola_shuff.shape[0])
rand_ind = np.random.choice(ind, size=int(rand_degree*cola_shuff.shape[0]), replace=False)

for ind in rand_ind:
    sent = SpaceTokenizer().tokenize(cola.iloc[ind][3])
    shuff_sent = shuffle_sentence(sent, shuffle_degree)
    cola_shuff.loc[ind, 3] = " ".join(shuff_sent)
    cola_shuff.loc[ind, 1] = 0

cola_shuff.to_csv(path_or_buf="glue_data/CoLA/dev_shuff.tsv", sep='\t', header=False, index=False)
rand_ind = np.random.choice(ind, size=int(rand_degree*cola_shuff.shape[0]), replace=False)

for ind in rand_ind:
    r = np.random.randint(0, cola.shape[0])
    sent = SpaceTokenizer().tokenize(cola.iloc[r][3])
    orig_sent = SpaceTokenizer().tokenize(cola.iloc[ind][3])
    shuff_sent = add_shuffle_sentence(orig_sent, sent, shuffle_degree)
    cola_add_shuff.loc[ind, 3] = " ".join(shuff_sent)
    cola_add_shuff.loc[ind, 1] = 0

cola_add_shuff.to_csv(path_or_buf="glue_data/CoLA/dev_add_shuff.tsv", sep='\t', header=False, index=False)