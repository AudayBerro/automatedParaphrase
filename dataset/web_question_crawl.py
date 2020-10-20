#this code crawl the WebQuestion datasets github repo
import requests
import json
import urllib
import unidecode # convert unicode string to ascii string
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import statistics
import time


def get_smooth(sentence_bleu,hyp,ref,weights, smoothing_function):
    """
    This function return the BLEU-Score
    :param sentence_bleu: nltk.translate.bleu_score.sentence_bleu 
    :param hyp: hypothesis sentences, list(str)
    :param ref: reference sentences, list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on, list of float
    :param smoothing_function: SmoothingFunction
    :return BLEU-score
    """
    return sentence_bleu(ref, hyp, weights = weights, smoothing_function = smoothing_function)

def get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function):
    """
    This function return the dataset BLEU score using median
    :param dataset: the data to calculate the BLUE_Score
    :param sentence_bleu: nltk.translate.bleu_score.sentence_bleu
    :param weights: weights for unigrams, bigrams, trigrams and so on, list of float
    :param smoothing_function: SmoothingFunction
    :return median of a list of BLEU-score
    """
    data_set_bleu_score = [] #bleu score of all dataset

    for k,v in dataset.items():
        reference = [k.lower().split(" ")]
        utterance_bleu_score = [] # current utterance bleu_Score = average_paraphrase_bleu_score
        for cand in v:
            candidate = cand.lower().split(" ")

            parpahrase_bleu_Score = get_smooth(sentence_bleu,candidate,reference,weights,smoothing_function)

            utterance_bleu_score.append(parpahrase_bleu_Score)
        
        if len(utterance_bleu_score) > 0:
            utterance_bleu_score = statistics.median(utterance_bleu_score)
        data_set_bleu_score.append(utterance_bleu_score)
    bleu = statistics.median(data_set_bleu_score)

    return bleu


def get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function):
    """
    This function return the dataset BLEU score using mean
    :param dataset: the data to calculate the BLUE_Score
    :param sentence_bleu: nltk.translate.bleu_score.sentence_bleu
    :param weights: weights for unigrams, bigrams, trigrams and so on, list of float
    :param smoothing_function: SmoothingFunction
    :return mean of a list of BLEU-Score 
    """
    data_set_bleu_score = 0 #bleu score of all dataset

    for k,v in dataset.items():
        reference = [k.lower().split(" ")]
        utterance_bleu_score = 0 # current utterance bleu_Score = average_paraphrase_bleu_score
        for cand in v:
            candidate = cand.lower().split(" ")
            
            paraphrase_bleu_score = get_smooth(sentence_bleu,candidate,reference,weights,smoothing_function)
            
            utterance_bleu_score += paraphrase_bleu_score
        
        if utterance_bleu_score > 0:
            utterance_bleu_score = utterance_bleu_score / len(v)
        data_set_bleu_score += utterance_bleu_score

    bleu = data_set_bleu_score / len(dataset)
    return bleu

def get_individual_bleu_score(dataset,sentence_bleu,flag):
    """
    Compute Individual N-Gram Scores An individual N-gram score is the evaluation of just matching grams of a specific order, such as single words (1-gram) or word pairs (2-gram or bigram).
    :param dataset: the data to calculate the BLUE_Score
    :param sentence_bleu: nltk.translate.bleu_score.sentence_bleu
    :param flag: if flag = 0 call get_bleu_score_mean() else call get_bleu_score_median()
    :return BLEU-1,BLEU-2,BLEU-3,BLEU-4
    """
    cc = SmoothingFunction()

    if flag == 0: # compute BLEU-Score using median
        smoothing_function = cc.method2 # cc.method1, cc.method3 or cc.method4  
        
        #BLEU-1
        weights = (1,0,0,0) # for N-gram where N>4 => weights = (0, .. ,0,1,0, .. ,0) at N-position set to 1
        bleu1 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-2
        weights = (0,1,0,0)
        bleu2 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-3
        weights = (0,0,1,0)
        bleu3 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-4
        weights = (0,0,0,1)
        bleu4 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)


    else:
        smoothing_function = cc.method2 # cc.method1, cc.method3 or cc.method4  
        
        #BLEU-1
        weights = (1,0,0,0) # for N-gram where N>4 => weights = (0, .. ,0,1,0, .. ,0) at N-position set to 1
        bleu1 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-2
        weights = (0,1,0,0)
        bleu2 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-3
        weights = (0,0,1,0)
        bleu3 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-4
        weights = (0,0,0,1)
        bleu4 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)
    
    return bleu1,bleu2,bleu3,bleu4

def get_cumulative_bleu_score(dataset,sentence_bleu,flag):
    """
    Cumulative N-Gram Scores refer to the calculation of individual n-gram scores at all orders from 1 to n and weighting them by calculating the weighted geometric mean.
    :param dataset: the data to calculate the BLUE_Score
    :param sentence_bleu: nltk.translate.bleu_score.sentence_bleu
    :param flag: if flag = 0 call get_bleu_score_mean() else call get_bleu_score_median()
    :return cumulative BLEU-2,BLEU-3,BLEU-4
    """
    cc = SmoothingFunction()

    if flag == 0: # compute BLEU-Score using median
        smoothing_function = cc.method2 # cc.method1, cc.method3 or cc.method4  
        
        #BLEU-2
        weights = (0.5,0.5,0,0) # for N-gram where N>4 => weights = (0, .. ,0,1,0, .. ,0) at N-position set to 1
        bleu2 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-3
        weights = (0.3,0.3,0.3,0)
        bleu3 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-4
        weights = (0.25,0.25,0.25,0.25)
        bleu4 = get_bleu_score_mean(dataset,sentence_bleu,weights,smoothing_function)


    else:
        smoothing_function = cc.method2 # cc.method1, cc.method3 or cc.method4  
        
        #BLEU-2
        weights = (0.5,0.5,0,0) # for N-gram where N>4 => weights = (0, .. ,0,1,0, .. ,0) at N-position set to 1
        bleu2 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-3
        weights = (0.3,0.3,0.3,0)
        bleu3 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)

        #BLEU-4
        weights = (0.25,0.25,0.25,0.25)
        bleu4 = get_bleu_score_median(dataset,sentence_bleu,weights,smoothing_function)
    
    return bleu2,bleu3,bleu4

def write_result(data,name):
    """ 
    Write output in folder
    :param data: data to preserve in folder
    """
    f = open("./dataset/"+name,"a")

    f.write("question\tnumber of paraphrases\tparaphrases\n")
    for k,v in data.items():
        f.write(k+"\t"+str(len(v)))
        for e in v:
            f.write("\t"+e)
        f.write("\n")

    f.close()

def crawl_data():
    """
    Crawl GraphQuestions repository
    :return crawled data in a form of Python dictionary where Key is the utterance and value is a list of paraphrases
    """
    url = "https://raw.githubusercontent.com/ysu1989/GraphQuestions/master/freebase13/graphquestions.testing.json"
    req = requests.get(url)
    data = json.loads(req.content)


    result = {}
    key = str(data[0]['question'])
    qid = data[0]['graph_query']['nodes'][0]['friendly_name']

    result[key] = []

    for t in data[1:]:
        if t['graph_query']['nodes'][0]['friendly_name'] == qid:
            result[key].append(unidecode.unidecode(t['question']))
        else:
            key = unidecode.unidecode(t['question'])
            qid = t['graph_query']['nodes'][0]['friendly_name']
            result[key] = []
    
    return result

def main():
    """ Crawl GraphQuestions repo and compute Individual and Cumulative BLEU-Score """
    
    result = crawl_data()

    bleu = sentence_bleu
    print("============================================================")
    print("  Compute Individual N-gram BLEU-Score using mean: ")
    b1,b2,b3,b4 = get_individual_bleu_score(result,bleu,0)
    print("\tIndividual 1-gram: ",b1)
    print("\tIndividual 2-gram: ",b2)
    print("\tIndividual 3-gram: ",b3)
    print("\tIndividual 4-gram: ",b4)

    print("  Compute Individual N-gram BLEU-Score using median: ")
    b1,b2,b3,b4 = get_individual_bleu_score(result,bleu,1)
    print("\tIndividual 1-gram: ",b1)
    print("\tIndividual 2-gram: ",b2)
    print("\tIndividual 3-gram: ",b3)
    print("\tIndividual 4-gram: ",b4)

    print("============================================================")
    print("  Compute Cumulative N-gram BLEU-Score using mean: ")
    b2,b3,b4 = get_cumulative_bleu_score(result,bleu,0)
    print("\tIndividual 2-gram: ",b2)
    print("\tIndividual 3-gram: ",b3)
    print("\tIndividual 4-gram: ",b4)

    print("  Compute Cumulative N-gram BLEU-Score using median: ")
    b2,b3,b4 = get_cumulative_bleu_score(result,bleu,1)
    print("\tIndividual 2-gram: ",b2)
    print("\tIndividual 3-gram: ",b3)
    print("\tIndividual 4-gram: ",b4)
    print("============================================================")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = "result-"+timestr+".csv"
    print("Save data in \""+file_name+"\"")
    write_result(result,file_name)

if __name__ == "__main__":
    main()