from embedding_as_service.text.encode import Encoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

""" Word embedding based on embedding-as-service https://github.com/amansrivastava17/embedding-as-service  """

def get_similarity(vector1,vector2):
    """
    Apply cosine similarity on embeddings vector
    :param vector1: embedding of the first sentence 
    :param model: embedding of the second sentence 
    :return the cosine similarity between vecotr1 and vector2
    """
    # a = vector1[0].reshape(1,-1)
    # b = vector2[0].reshape(1,-1)

    cos_lib = cosine_similarity(vector1,vector2)
    print(cos_lib)
    return np.asscalar(cos_lib)#convert numpy array to float

def arccos_similarity(vector1,vector2):
    """
    Return similarity between embeddings vecor. sim(u,v) = 1 - (arcos[(u.v)/(||u||.||v||)] / pi)
    :param vector1: embedding of the first sentence 
    :param model: embedding of the second sentence 
    :return similarity between vecotr1 and vector2
    """
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    dot_uv = np.ndarray.dot(vector1,vector2)
    norm_uv = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    arcosine = np.arccos(dot_uv/norm_uv)/np.pi
    return 1-arcosine

#def filtering(pool,embedding='use', model='use_transformer_large', max_seq_length=128,pooling='reduce_mean'):
def filtering(pool,embedding='bert', model='bert_base_cased', max_seq_length=128,pooling='reduce_mean'):
    """
    Remove paraphrases that are not semantically equivalent to the initial expression 
    :param pool: a Python dictionary, Key: initial expression, value: set of paraphrases
    :param embedding: str embedding method to be used, check Embedding column here e.g: bert, elmo, use, glove, etc
    :param model: str Model to be used for mentioned embedding e.g bert_base_cased, bert_base_uncased, use_transformer_large, elmo_bi_lm, etc 
    :param max_seq_length: int maximum length of a sequence after tokenization, default is 256
    :param pooling: Pooling strategy to apply e.g. reduce_mean, reduce_max, etc 
    :return a Python dictionary where not semantically equivalent paraphrases are removed
    """
    en = Encoder(embedding, model, max_seq_length) #load encoder
    result = dict()
    for key,value in pool.items():
        vector1 = en.encode([key], pooling)
        # a = vector1.reshape(1,-1)
        paraphrases = []
        for candidate in value:
            vector2 = en.encode([candidate], pooling)
            # b = vector2.reshape(1,-1)
            # cos_sim = arccos_similarity(vector1,vector2)
            cos_sim = get_similarity(vector1,vector2)
            print(cos_sim)
            if cos_sim > 0.5:
                paraphrases.append(candidate)
        result[key] = paraphrases
    return result

if __name__ == "__main__":
    pool = {'book a flight from lyon to sydney? ': ['book a flight lyon', 'lyon book a flight to sydney?', 'to bible flights from lyon to sydney?']}
    a = filtering(pool)
    print(a)