from nltk.translate.chrf_score import sentence_chrf
import csv
import statistics

def apply_cut_off(pool,cut_off):
    """
    This function extract the [cut_off] top highest semantically related paraphrases
    :param pool: python dictionary, key is the initial utterance and value is a list of tuples. Tuples(paraphrase, BERT embeddong cosine similarity score)
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return ordred Python dictionary
    """

    if cut_off == 0:
        return pool
    else:
        result = {}
        for k,v in pool.items():
            if len(v) <= cut_off: # if list of paraphrases [v] contain less than [cut_off]-element, add all element
                result[k]=v
            else:
                result[k]=v[:cut_off]
        return result

def get_chrf_score(sentence_chrf,hyp,ref):
    """
    This function return the chrf-Score
    :param sentence_chrf: nltk.translate.chrf_score.sentence_chrf 
    :param hyp: hypothesis sentences, list(str)
    :param ref: reference sentences, list(list(str))
    :return chrf-score
    """
    return sentence_chrf(ref, hyp)

def get_chrf_score_median(dataset,sentence_chrf):
    """
    This function return the dataset chrf score using median
    :param dataset: the data to calculate the GLUE_Score
    :param sentence_chrf: nltk.translate.chrf_score.sentence_chrf
    :return median of a list of chrf-score
    """
    data_set_chrf_score = [] #chrf score of all dataset

    for k,v in dataset.items():
        reference = k.lower().split(" ")
        utterance_chrf_score = [] # current utterance chrf_Score = average_paraphrase_chrf_score
        if len(v) > 0 :
          for cand in v:
              candidate = cand.lower().split(" ")

              parpahrase_chrf_Score = get_chrf_score(sentence_chrf,candidate,reference)

              utterance_chrf_score.append(parpahrase_chrf_Score)
          utterance_chrf_score = statistics.median(utterance_chrf_score)
          data_set_chrf_score.append(utterance_chrf_score)
    chrf = statistics.median(data_set_chrf_score)

    return chrf


def get_chrf_score_mean(dataset,sentence_chrf):
    """
    This function return the dataset chrf score using mean
    :param dataset: the data to calculate the GLUE_Score
    :param sentence_chrf: nltk.translate.chrf_score.sentence_chrf
    :return mean of a list of chrf-Score 
    """
    score = 0 #chrf score of all dataset

    for k,v in dataset.items():
        reference = k.lower().split(" ")
        utterance_chrf_score = 0 # current utterance chrf_Score = average_paraphrase_chrf_score
        for cand in v:
            candidate = cand.lower().split(" ")
            
            paraphrase_chrf_score = get_chrf_score(sentence_chrf,candidate,reference)
            
            utterance_chrf_score += paraphrase_chrf_score
        
        if utterance_chrf_score > 0:
            utterance_chrf_score = utterance_chrf_score / len(v)
        score += utterance_chrf_score

    chrf = score / len(dataset)
    return chrf

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

def get_score(data,cut_off):
    chrf = sentence_chrf
    # data = read_data("output.csv")
    print("\n\t============================================================")
    print("\t                  Cut_off parpameter = ",cut_off,"            ")
    print("\t============================================================")
    data = apply_cut_off(data,cut_off)
    print("\t  Compute chrf-Score using mean: ", end="")
    chrf_score = get_chrf_score_mean(data,chrf)
    print(" ",chrf_score)

    print("\t  Compute chrf-Score using median: ", end="")
    chrf_score = get_chrf_score_median(data,chrf)
    print(" ",chrf_score)

def main(data,cut_off):
    """
    Compute chrf-Score for data
    :param data: python dictionary key initial utterance, value list of parpahrases
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    """

    if cut_off == 0:
        cut_off = [0,3,5,10,20,50,100]
        
        for cut in cut_off:
            get_score(data,cut)
    else:
        get_score(data,cut_off)

if __name__ == "__main__":
    main()

