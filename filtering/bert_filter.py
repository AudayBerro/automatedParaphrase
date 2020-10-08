from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

""" BERT Word embeddings for similarity """

def concatenate_output(outputs):
    """
    Concatenate the last four hidden layer output
    :param outputs: BERT model output
    :return a vector formed by the concatenation of the output of the last four hidden layer
    """
    
    token_vecs_cat = []
    # For each token in the sentence...
    for token in outputs[0]:
        # Concatenate the vectors (that is, append them together) from the last four layers
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    return token_vecs_cat

def token_vector_sum(token_embeddings): 
    """
    Summing togheter all token embedding
    :param token_embeddings: a list of token embeddings
    :return a vector formed by the summation of all token embeddings
    """

    sum_vec = torch.sum(token_embeddings,dim=0)
    return sum_vec

def token_vector_mean(token_embeddings):#return a sentence vecotr as the mean average of all token vector
    """
    Get the mean average of all token embedding
    :param token_embeddings: a list of token embeddings
    :return a vector formed by the mean average of all token embeddings
    """

    # sum_vec = torch.sum(token_embeddings[1:-2],dim=0)#sum of all token except the first([CLS]=0) and the last token([SEP] = -1)
    sentence_embedding = torch.mean(token_embeddings, dim=0)
    return sentence_embedding

def get_encoded_layers(utterance,model,tokenizer):#huggingface transformers library embeddings
    """
    Get BERT embedding of a sentence using the Huggingface transformers
    :param utterance: the sentence to embed
    :param model: Huggingface transformers BERT model
    :param tokenizer: Huggingface transformers BERT Tokenizer
    :return Huggingface transform BERT model output consisting of tuple(torch.FloatTensor) comprising various elements depending on the configuration (BertConfig) and inputs
    """
    input_ids = torch.tensor(tokenizer.encode(utterance, add_special_tokens=True)).unsqueeze(0)

    with torch.no_grad():
        encoded_layers = model(input_ids)
        return encoded_layers
        

def get_similarity(vector1,vector2):
    """
    Apply cosine similarity on BERT embeddings vector
    :param vector1: embedding of the first sentence 
    :param model: embedding of the second sentence 
    :return the cosine similarity between vecotr1 and vector2
    """
    a = vector1[0].reshape(1,-1)
    a = a.detach().numpy()
    b = vector2[0].reshape(1,-1)
    b = b.detach().numpy()

    cos_lib = cosine_similarity(a,b)
    return np.asscalar(cos_lib)#convert numpy array to float


def bert_selection(pool):#ebeddings using Huggingface trandformers library
    """
    Remove paraphrases that are not semantically equivalent to the initial expression and duplicate(filtering+deduplication)
    :param pool: a Python dictionary, Key is the initial expression, value is a set of paraphrases
    :return a Python dictionary where not semantically equivalent paraphrases and duplicate are removed
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # to get hidden layer in the output uncomment the following above code line, see https://huggingface.co/transformers/model_doc/bert.html for details
    # model = BertForPreTraining.from_pretrained('bert-base-uncased')
    # config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    # model = BertModel.from_pretrained("bert-base-uncased", config=config) #return hidden state in the output
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    # input_ids = torch.tensor(tokenizer.encode("After stealing money from the bank vault, the bank robber was seen", add_special_tokens=True)).unsqueeze(0)
    result = dict()
    for key,value in pool.items():
        a = get_encoded_layers(key,model,tokenizer)
        vector1 = concatenate_output(a)

        # token_embeddings = a[0][0]
        # vectora = token_vector_sum(a[0][0])
        # vectora2 = token_vector_mean(a[0][0])
        paraphrases = []
        for candidate in value:
            b = get_encoded_layers(candidate,model,tokenizer)
            vector2 = concatenate_output(b)

            # token_embeddings = b[0][0]
            # vectorb = token_vector_sum(b[0][0])
            # vectorb2 = token_vector_mean(b[0][0])
            cos_sim = get_similarity(vector1,vector2)
            # cos_sim2 = ukplab_similarity(vectora,vectorb)
            if cos_sim > 0.5 and cos_sim <= 0.95:
                paraphrases.append((candidate,cos_sim))
        result[key] = paraphrases
    return result

def bert_filtering(pool):#ebeddings using Huggingface trandformers library
    """
    Remove paraphrases that are not semantically equivalent to the initial expression 
    :param pool: a Python dictionary, Key is the initial expression, value is a set of paraphrases
    :return a Python dictionary where not semantically equivalent paraphrases are removed
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # to get hidden layer in the output uncomment the following above code line, see https://huggingface.co/transformers/model_doc/bert.html for details
    # model = BertForPreTraining.from_pretrained('bert-base-uncased')
    # config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    # model = BertModel.from_pretrained("bert-base-uncased", config=config) #return hidden state in the output
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    # input_ids = torch.tensor(tokenizer.encode("After stealing money from the bank vault, the bank robber was seen", add_special_tokens=True)).unsqueeze(0)
    result = dict()
    for key,value in pool.items():
        a = get_encoded_layers(key,model,tokenizer)
        vector1 = concatenate_output(a)

        # token_embeddings = a[0][0]
        # vectora = token_vector_sum(a[0][0])
        # vectora2 = token_vector_mean(a[0][0])
        paraphrases = []
        for candidate in value:
            b = get_encoded_layers(candidate,model,tokenizer)
            vector2 = concatenate_output(b)

            # token_embeddings = b[0][0]
            # vectorb = token_vector_sum(b[0][0])
            # vectorb2 = token_vector_mean(b[0][0])
            cos_sim = get_similarity(vector1,vector2)
            # cos_sim2 = ukplab_similarity(vectora,vectorb)
            if cos_sim > 0.5:
                paraphrases.append(candidate)
        result[key] = paraphrases
    return result

def bert_deduplication(pool):#remove deduplicate paraphrases
    """
    Remove paraphrases that are duplication of the initial expression 
    :param pool: a Python dictionary, Key is the initial expression, value is a set of paraphrases
    :return a Python dictionary where duplicate paraphrases are removed
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    result = dict()
    for key,value in pool.items():
        a = get_encoded_layers(key,model,tokenizer)
        vector1 = concatenate_output(a)

        # token_embeddings = a[0][0]
        # vectora = token_vector_sum(a[0][0])
        # vectora2 = token_vector_mean(a[0][0])
        paraphrases = []
        for candidate in value:
            b = get_encoded_layers(candidate,model,tokenizer)
            vector2 = concatenate_output(b)

            # token_embeddings = b[0][0]
            # vectorb = token_vector_sum(b[0][0])
            # vectorb2 = token_vector_mean(b[0][0])
            cos_sim = get_similarity(vector1,vector2)
            if cos_sim <= 0.95:
                paraphrases.append(candidate)
        result[key] = paraphrases
    return result


def ukplab_similarity(vector1,vector2):
    """
    Cosine similarity using UKPLab sentence-transformers library  embeddings vector
    :param vector1: UKPLab sentence-transformers embedding of sentence 1
    :param vector1: UKPLab sentence-transformers embedding of sentence 2
    :return cosine similarity between vector1 and vector2
    """
    a = vector1.reshape(1,-1)
    # b = embedding_output2.reshape(1,-1)
    b = vector2.reshape(1,-1)
    cos_lib = cosine_similarity(a,b)
    return np.asscalar(cos_lib)#convert numpy array to float

def get_embeddings(utterance,embedder):# embeddings using UKPLab sentence_transformers library
    """
    UKPLab sentence-transformers  sentence embeddings
    :param utterance: sentence to embed
    :param embedder: UKPLab sentence-transformers model
    :return sentence embedding using UKPLab sentence-transformers library
    """

    sentence_embeddings = embedder.encode(utterance)
    return sentence_embeddings

def ukplab_filtering(pool):#embeddings using UKPLab sentence_transformers library
    """
    Remove paraphrases that are not semantically equivalent to the initial expression using the UKPLab sentence-transformers library 
    :param pool: a Python dictionary, Key is the initial expression, value is a set of paraphrases
    :return a Python dictionary where not semantically equivalent paraphrases are removed
    """

    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    result = dict()
    for key,value in pool.items():
        a = get_embeddings(key,embedder)
        vector1 = a[0]
        print(key)
        paraphrases = []
        for candidate in value:
            b = get_embeddings(candidate,embedder)
            vector2 = b[0]
            cos_sim = ukplab_similarity(vector1,vector2)
            if cos_sim > 0.5:
                paraphrases.append(candidate)
                print(key,",",candidate,"= ",cos_sim)
        result[key] = paraphrases
    return result
