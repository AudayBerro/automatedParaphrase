from embedding_as_service.text.encode import Encoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

""" Compute semantic similarity using embedding-as-service https://github.com/amansrivastava17/embedding-as-service  """

def get_similarity(vector1,vector2):
    """
    Apply cosine similarity on embeddings vector
    :param vector1: embedding of the first sentence 
    :param model: embedding of the second sentence 
    :return the cosine similarity between vecotr1 and vector2
    """
    a = vector1[0].reshape(1,-1)
    b = vector2[0].reshape(1,-1)

    cos_lib = cosine_similarity(a,b)
    return np.asscalar(cos_lib)#convert numpy array to float

def main():
    en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=256)
    vecs = en.encode(texts=['how does COVID-10 spread','book a flight from lyon to sydney'], pooling='reduce_mean')
    # a = vecs[0].reshape(1,-1)
    # b = vecs[1].reshape(1,-1)
    cos_lib = get_similarity(vecs[0],vecs[1])
    print(cos_lib)

if __name__ == "__main__":
    main()