#this code compute the bleu score of candidate paraphrases

import statistics
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import csv


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
        if len(v) > 0 :
          for cand in v:
              candidate = cand.lower().split(" ")

              parpahrase_bleu_Score = get_smooth(sentence_bleu,candidate,reference,weights,smoothing_function)

              utterance_bleu_score.append(parpahrase_bleu_Score)
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
    score = 0 #bleu score of all dataset

    for k,v in dataset.items():
        reference = [k.lower().split(" ")]
        utterance_bleu_score = 0 # current utterance bleu_Score = average_paraphrase_bleu_score
        for cand in v:
            candidate = cand.lower().split(" ")
            
            paraphrase_bleu_score = get_smooth(sentence_bleu,candidate,reference,weights,smoothing_function)
            
            utterance_bleu_score += paraphrase_bleu_score
        
        if utterance_bleu_score > 0:
            utterance_bleu_score = utterance_bleu_score / len(v)
        score += utterance_bleu_score

    bleu = score / len(dataset)
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

##############################################################################################################
#data_set_bleu_score = 0 #bleu score of all dataset

def read_data(file_name):
  """
  Read csv file and convert to Python dictionary
  :param filename: csv file to read
  :return a python dictionary where key is utterance and value a list of paraphrases
  """
 
  import csv
  tsv_file = open(file_name)
  reader = csv.reader(tsv_file,delimiter="\t")
  d = {}
  for row in reader:
     if int(row[1]) > 2:
        d[row[0]]= row[2:]
  
  return d

def get_bleu_score(data,cut_off):
    """
    Compute BLEU-Score
    :param data: python dictionary key initial utterance, value list of parpahrases
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return a python list containing BLEU scores and comment to print or save in file
    """
    bleu = sentence_bleu
    # data = read_data("output.csv")
    result = []

    result.append("\t============================================================")
    result.append(f"\t                    BLEU score = {cut_off}")
    result.append("\t============================================================")
    data = apply_cut_off(data,cut_off)
    result.append("\t  Compute Individual N-gram BLEU-Score using mean: ")
    b1,b2,b3,b4 = get_individual_bleu_score(data,bleu,0)
    result.append(f"\t\tIndividual 1-gram: {b1}")
    result.append(f"\t\tIndividual 2-gram: {b2}")
    result.append(f"\t\tIndividual 3-gram: {b3}")
    result.append(f"\t\tIndividual 4-gram: {b4}")

    result.append(f"\t  Compute Individual N-gram BLEU-Score using median: ")
    b1,b2,b3,b4 = get_individual_bleu_score(data,bleu,1)
    result.append(f"\t\tIndividual 1-gram: {b1}")
    result.append(f"\t\tIndividual 2-gram: {b2}")
    result.append(f"\t\tIndividual 3-gram: {b3}")
    result.append(f"\t\tIndividual 4-gram: {b4}")

    result.append(f"\t============================================================")
    result.append(f"\t  Compute Cumulative N-gram BLEU-Score using mean: ")
    b2,b3,b4 = get_cumulative_bleu_score(data,bleu,0)
    result.append(f"\t\tIndividual 2-gram: {b2}")
    result.append(f"\t\tIndividual 3-gram: {b3}")
    result.append(f"\t\tIndividual 4-gram: {b4}")

    result.append(f"\t  Compute Cumulative N-gram BLEU-Score using median: ")
    b2,b3,b4 = get_cumulative_bleu_score(data,bleu,1)
    result.append(f"\t\tIndividual 2-gram: {b2}")
    result.append(f"\t\tIndividual 3-gram: {b3}")
    result.append(f"\t\tIndividual 4-gram: {b4}")

    return result

def main(data,cut_off):
    """
    Compute BLEU-Score for data
    :param data: python dictionary key initial utterance, value list of parpahrases
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return a python List of scores and strings to print or save in a file
    """

    result = []
    if cut_off == 0:
        cut_off = [0,3,5,10,20,50,100]
        
        for cut in cut_off:
            result.extend(get_bleu_score(data,cut))
    else:
        result.extend(get_bleu_score(data,cut_off))
    
    return result

if __name__ == "__main__":
    main()
