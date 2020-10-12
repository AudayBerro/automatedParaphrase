import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

""" Universal Sentence Encoder word embeddings """


def get_embedding(data):
    """
    Get Universal Sentence Encoder embeddings
    :param data: python dictionary on which to apply embedding, Key initial sentence and value is a set of paraphrases
    :return a python dictionary whre not equivalent paraphrase to initial sentence are removed
    """
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    response = dict()
    for key,value in data.items():
        key_embedding = embed([key]) #initial sentence USE embedding
        a=np.reshape(key_embedding,(1,-1))
        tmp = []
        for candidate in value:
            candidate_embedding = embed([candidate]) #candidate USE embedding
            b=np.reshape(candidate_embedding,(1,-1))
            cos_lib = cosine_similarity(a,b) #cosine similarity
            b = 0
            if cos_lib > 0.5:
                tmp.append(candidate)
            response[key]=tmp

    return response
