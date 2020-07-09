from embedding_as_service.text.encode import Encoder
from sklearn.metrics.pairwise import cosine_similarity

""" Compute semantic similarity using embedding-as-service https://github.com/amansrivastava17/embedding-as-service  """

def main():
    en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=256)
    vecs = en.encode(texts=['how does COVID-10 spread','book a flight from lyon to sydney'], pooling='reduce_mean')
    a = vecs[0].reshape(1,-1)
    b = vecs[1].reshape(1,-1)
    cos_lib = cosine_similarity(a,b)
    print(cos_lib)

if __name__ == "__main__":
    main()