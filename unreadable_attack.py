import editdistance
import random
import numpy as np

def shuffle_sentence(sent, shuffle_degree=0.65):
    edit_dist = len(sent)*shuffle_degree
    shuffle_sent = random.sample(sent.tolist(), len(sent))
    while editdistance.eval(sent, shuffle_sent) < edit_dist:
        shuffle_sent = random.sample(sent.tolist(), len(sent))
    return np.array(shuffle_sent)


def add_shuffle_sentence(sent, sent_to_be_shuff, shuffle_degree=0.65):
    shuffle_sent = shuffle_sentence(sent_to_be_shuff, shuffle_degree)
    return np.concatenate((sent, shuffle_sent))


# def paraphrase_sentence(sent):
	