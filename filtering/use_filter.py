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
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    response = dict()
    for key,value in data.items():
        with tf.compat.v1.Session() as session:
            session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
            key_embedding = embed([key]) #initial sentence USE embedding
            vector1 = session.run(key_embedding)
            a=np.reshape(vector1,(1,-1))
            tmp = []
            for candidate in value:
                candidate_embedding = embed([candidate]) #candidate USE embedding
                vector2 = session.run(candidate_embedding)
                b=np.reshape(vector2,(1,-1))
                cos_lib = cosine_similarity(a,b) #cosine similarity
                b = 0
                if cos_lib > 0.5:
                    tmp.append(candidate)
            session.close()
            response[key]=tmp

    return response