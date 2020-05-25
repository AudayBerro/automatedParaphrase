import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

""" Universal Sentence Encoder word embeddings """



def get_embedding(data):
    """
    Get Universal Sentence Encoder embeddings
    :param data: python dictionary on which to apply embedding, Key initial sentence and value is a set of paraphrases
    :return a python dictionary whre not equivalent paraphrase to initial sentence are removed
    """
    disable_eager_execution() # to avoid EagerTensor RuntimeError
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    response = dict()
    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        for key,value in data.items():
            key_embedding = embed([key]) #initial sentence USE embedding
            vector1 = session.run(key_embedding)
            tmp = []
            for candidate in value:
                candidate_embedding = embed([candidate]) #candidate USE embedding
                vector2 = session.run(candidate_embedding)
                
                a=np.reshape(vector1,(1,-1))
                b=np.reshape(vector2,(1,-1))
                cos_lib = cosine_similarity(a,b) #cosine similarity
                if cos_lib > 0.5:
                    tmp.append(candidate)
            response[key]=tmp

    return response