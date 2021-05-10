import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

""" Universal Sentence Encoder word embeddings """

def pr_gray(msg):
    """ Pring msg in gray color font"""
    print("\033[7m{}\033[00m" .format(msg))

def pr_green(msg):
    """ Pring msg in green color font"""
    print("\033[92m{}\033[00m" .format(msg))

def load_model(model_name="https://tfhub.dev/google/universal-sentence-encoder-large/5"):
    """
    Load Universal Sentence Encoder model
    :param model_name: name of the USE model to load
    :return an USE model
    """
    pr_gray("\nLoad Universal Sentence Encoder model:")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    pr_green("... done")
    return model


def get_embedding(data,embed):
    """
    Get Universal Sentence Encoder embeddings
    :param data: python dictionary on which to apply embedding, Key initial sentence and value is a set of paraphrases
    :param embed: Universal Sentence Encoder model instance
    :return a python dictionary whre not equivalent paraphrase to initial sentence are removed
    """

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

def test():
    print("Load USE ")
    embed = load_model("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    print("... done")

    d = {'how does covid-19 spread':["how does it spread","book a flight from lyon to sydney",'i feel cold']}
    r = get_embedding(d,embed)
    print(r)

if __name__ == '__main__':
    test()