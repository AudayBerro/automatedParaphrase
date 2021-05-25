# Metrics to measure data diversity TTR, PINC and DIV
from nltk.stem import WordNetLemmatizer
import nltk
import re

def pre_process(text):
    """
    Lowercase and lemmatize a sentence
    :param text: sentence to preprocess
    :return preprocessed sentence
    """
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)
    return re.sub(r'[^\w\s]', ' ', text).lower()


def remove_cosine_score(data):
    """
    Remove cosine similarity from value list
    :param data: Python dictionary, key:initial utterance, value: list of tuple paraphrases, tuples:(paraphrase,BERT embedding cosine similarity with key)
    :return Python dictionary key:initial utterance, value list of paraphrases without cosine similarity score
    """

    response = {}
    for k,v in data.items():
        if data[k]:
            response[k] = []
            for t in v:
                response[k].append(t[0])
    
    return response

def apply_cut_off(pool,cut_off):
    """
    This function extract the [cut_off] top highest semantically related paraphrases
    :param pool: python dictionary, key is the initial utterance and value is a list of tuples. Tuples(paraphrase, BERT embeddong cosine similarity score)
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return ordred Python dictionary
    """

    if cut_off <= 0:
        return pool
    else:
        result = {}
        for k,v in pool.items():
            if len(v) <= cut_off: # if list of paraphrases [v] contain less than [cut_off]-element, add all element
                result[k]=v
            else:
                result[k]=v[:cut_off]
        return result

def get_ttr_score(data):
    """
    Type-Token-Ratio calculates the ratio of unique words to the total number of words in the utterances
    :param data: Dataset to be measured in terms of TTR. Python dictionary, key: initial utterance - value: list of paraphrases
    :return Mean TTR score of a dictionary of paraphrases
    """
    # ret = {}
    ttrs = []
    total_words = 0
    total_unique_words = 0
    mean_ttr = 0
    counter = 0
    for expr in data.keys():
        vset = set()
        tlen = 0
        for p in data[expr]:
            p = pre_process(p)
            tokens = p.split(" ") # can use SpaCy or NLTK tokenizer insteasd of splitting
            tlen += len(tokens) 
            vset = vset.union(set(tokens))
        counter += 1
        total_words += tlen
        total_unique_words += len(vset)
        if tlen != 0:
            mean_ttr += len(vset) / tlen
            ttrs.append(len(vset) / tlen)
            # ret[expr] = {'vocabulary size': len(vset), 'total_num_of_words': tlen, 'TTR': len(vset) / tlen}

    return {"Total Words": total_words,
            "Total Unique Words": total_unique_words,
            "Mean TTR": mean_ttr / counter,
            "All TTRs": ttrs}

def pinc(source, paraphrase, n=4):
    """
    Paraphrase In N-gram Changes measures the percentage of n-gram changes between the initial utterance (a) and a collected utterance (b)
    :param source: sentence a
    :param paraphrase: sentence b
    :param n: n-grams, default n=4 e.g a=[1,2,3] b=[2,3,4] => 2-grams of a: {(1,2),(2,3)}
    :return PINC score between sentence (a) and sentence (b)
    """
    sum = 0
    index = 0
    for i in range(1, n + 1):
        s = set(nltk.ngrams(source, i,pad_left = False, pad_right = False))
        p = set(nltk.ngrams(paraphrase, i,pad_left = False, pad_right = False))
        if s and p:
            index += 1
            intersection = s.intersection(p)
            sum += 1 - len(intersection) / len(p)

    if index == 0:
        return 0

    return sum / index

def get_pinc_score(data):
    """
    Average of the PINC scores over all collected utterances. Paper: Collecting Highly Parallel Data for Paraphrase Evaluation
    :param data: Dataset to be measured in terms of PINC. Python dictionary, key: initial utterance - value: list of paraphrases
    :return Mean PINC score of a dictionary of paraphrases
    """
    index = 0
    total_pinc = 0
    pincs = []
    expr_by_pinc = []
    for expr in data:
        tokens_1 = pre_process(expr).split(" ")
        expr_pinc = 0
        for p in data[expr]:
            p = pre_process(p)
            tokens_2 = p.split(" ")
            expr_pinc += pinc(tokens_1, tokens_2)
            index += 1
        expr_pinc = expr_pinc / len(data[expr])
        pincs.append(expr_pinc)
        expr_by_pinc.append((expr, expr_pinc))
        total_pinc += expr_pinc
    expr_by_pinc.sort(key=lambda t: -t[1])
    return {"Mean PINC": total_pinc / len(data),
            "All PINCs": pincs,
            "Expr by Pinc": expr_by_pinc}


def jaccard_index(source, paraphrase, n=3):
    """ 
    Calculate the reverse of the mean Jaccard Index between the sentences’ n-grams sets to represent the semantic distances between the two sentences
    :param source: sentence a
    :param paraphrase: sentence b
    :param n: n-grams, default n=3 as set by author papers in their experiments e.g a=[1,2,3] b=[2,3,4] => 2-grams of a: {(1,2),(2,3)}
    :return reverse of the mean Jaccard Index between sentence a and sentence b
    """
    sum = 0
    for i in range(1, n + 1):
        s = set(nltk.ngrams(source, i,pad_left = False, pad_right = False))
        p = set(nltk.ngrams(paraphrase, i,pad_left = False, pad_right = False))
        if s and p:
            intersection = s.intersection(p) # intersection between s and p
            p = s.union(p) # union between s and p
            jaccard_index = len(intersection) / len(p) # The Jaccard index, also known as Intersection over Union
            sum += jaccard_index
    
    #return reverse of the mean of Jaccard index
    return 1 - sum/n

def get_div_score(data):
    """
    Papers: "Data Collection for a Production Dialogue System: A Clinc Perspective" and "Outlier Detection for Improved Data Quality and Diversity in Dialog Systems"
    Compute diversity as the average jaccard_index distance between all sentence pairs. Paper: Data Collection for Dialogue System: A Startup Perspective
    :param data: Dataset to be measured in terms of diversity. Python dictionary, key: initial utterance - value: list of paraphrases
    :return Diversity score of a dictionary of paraphrases
    """

    total_d = 0
    d_list = []

    for expr in data:
        local_d = 0
        index = 0
        for i, ps in enumerate(data[expr]):
            tokens_1 = pre_process(ps).split(" ") # tokenize ps(whitespace as delimeter) - can use Spacy or NLTK tokenizer
            tokens_1 = list(filter(None,tokens_1)) # remove empty string from list of tokens and convert filter object to list
            for j, p in enumerate(data[expr]):
                if j != i:
                    tokens_2 = pre_process(p).split(" ")
                    tokens_2 = list(filter(None,tokens_2))
                    local_d += jaccard_index(tokens_1, tokens_2) # jaccard_index(tokens_1, tokens_2,4) # to compute 4-gram DIV
                    index += 1

        if index != 0:
            local_d = local_d / index
            d_list.append(local_d)
            total_d += local_d
    
    return {"Diversity": total_d / len(data)}

def get_scores(data,cut_off):
    """
    Compute PINC DIV TTR
    :param data: Dataset to be measured in terms of diversity. Python dictionary, key: initial utterance - value: list of paraphrases
    :param cut_off: cut_off parameter
    :return Mean-TTR, Mean-PINC and DIV scores
    """
    
    #data = remove_cosine_score(data)
    result = {}
    if cut_off == 0:
        cut_off = [3,5,10,20]
        
        for cut in cut_off:
           tmp_data = apply_cut_off(data,cut)
           ttr_score = get_ttr_score(tmp_data)
           pinc_score = get_pinc_score(tmp_data)
           div_score = get_div_score(tmp_data)
           result[cut] = [ttr_score,pinc_score,div_score]
        
        #    print("\n============================================================")
        #    print("                  Cut_off parpameter = ",cut,"            ")
        #    print("============================================================")
        #    print("\tMean TTR: "+str(ttr_score["Mean TTR"]))
        #    print("\tMean PINC: "+str(pinc_score["Mean PINC"]))
        #    print("\tDiversity: "+str(div_score['Diversity']))
    else:
        data = apply_cut_off(data,cut_off)
        ttr_score = get_ttr_score(data)
        pinc_score = get_pinc_score(data)
        div_score = get_div_score(data)
        result[cut_off] = [ttr_score,pinc_score,div_score]
        # print("\n============================================================")
        # print("                  Cut_off parpameter = ",cut_off,"            ")
        # print("============================================================")
        # print("\tMean TTR: "+str(ttr_score["Mean TTR"]))
        # print("\tMean PINC: "+str(pinc_score["Mean PINC"]))
        # print("\tDiversity: "+str(div_score['Diversity']))
    return result

def main(data,cut_off):
    """
    Compute PINC DIV TTR
    :param data: Dataset to be measured in terms of diversity. Python dictionary, key: initial utterance - value: list of paraphrases
    :param cut_off: cut_off parameter
    :return Mean-TTR, Mean-PINC and DIV scores
    """

    diversity_score = get_scores(data,cut_off)
    result = []
    for k,v in diversity_score.items():
        result.append("\t============================================================")
        result.append(f"\t                  Cut_off parpameter = {k}")
        result.append("\t============================================================")
        result.append(f"\t\tMean TTR: {v[0]['Mean TTR']}")
        result.append(f"\t\tMean PINC: {v[1]['Mean PINC']}")
        result.append(f"\t\tDiversity: {v[2]['Diversity']}")
    
    return result