from nltk.translate.gleu_score import sentence_gleu
import csv
import statistics

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

def get_gleu_score(sentence_gleu,hyp,ref):
    """
    This function return the gleu-Score
    :param sentence_gleu: nltk.translate.gleu_score.sentence_gleu 
    :param hyp: hypothesis sentences, list(str)
    :param ref: reference sentences, list(list(str))
    :return gleu-score
    """
    return sentence_gleu(ref, hyp)

def get_gleu_score_median(dataset,sentence_gleu):
    """
    This function return the dataset gleu score using median
    :param dataset: the data to calculate the GLUE_Score
    :param sentence_gleu: nltk.translate.gleu_score.sentence_gleu
    :return median of a list of gleu-score
    """
    data_set_gleu_score = [] #gleu score of all dataset

    for k,v in dataset.items():
        reference = [k.lower().split(" ")]
        utterance_gleu_score = [] # current utterance gleu_Score = average_paraphrase_gleu_score
        if len(v) > 0 :
          for cand in v:
              candidate = cand.lower().split(" ")

              parpahrase_gleu_Score = get_gleu_score(sentence_gleu,candidate,reference)

              utterance_gleu_score.append(parpahrase_gleu_Score)
          utterance_gleu_score = statistics.median(utterance_gleu_score)
          data_set_gleu_score.append(utterance_gleu_score)
    gleu = statistics.median(data_set_gleu_score)

    return gleu


def get_gleu_score_mean(dataset,sentence_gleu):
    """
    This function return the dataset gleu score using mean
    :param dataset: the data to calculate the GLUE_Score
    :param sentence_gleu: nltk.translate.gleu_score.sentence_gleu
    :return mean of a list of gleu-Score 
    """
    score = 0 #gleu score of all dataset

    for k,v in dataset.items():
        reference = [k.lower().split(" ")]
        utterance_gleu_score = 0 # current utterance gleu_Score = average_paraphrase_gleu_score
        for cand in v:
            candidate = cand.lower().split(" ")
            
            paraphrase_gleu_score = get_gleu_score(sentence_gleu,candidate,reference)
            
            utterance_gleu_score += paraphrase_gleu_score
        
        if utterance_gleu_score > 0:
            utterance_gleu_score = utterance_gleu_score / len(v)
        score += utterance_gleu_score

    gleu = score / len(dataset)
    return gleu

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
    """
    Compute Google-BLEU~(GLEU) Score
    :param data: python dictionary key initial utterance, value list of parpahrases
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return a python list containing GLEU scores and comment to print or save in file
    """
    gleu = sentence_gleu
    # data = read_data("output.csv")
    result = []

    result.append("\n\t============================================================")
    result.append(f"\t                  Cut_off parpameter = {cut_off}")
    result.append("\t============================================================")
    data = apply_cut_off(data,cut_off)
    
    gleu_score = get_gleu_score_mean(data,gleu)
    result.append(f"\t  Compute GLEU-Score using mean: {gleu_score}")


    gleu_score = get_gleu_score_median(data,gleu)
    result.append(f"\t  Compute GLEU-Score using median: {gleu_score}")

    return result

def main(data,cut_off):
    """
    Compute gleu-Score for data
    :param data: python dictionary key initial utterance, value list of parpahrases
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return a python List of scores and strings to print or save in a file
    """
    result = []

    if cut_off == 0:
        cut_off = [0,3,5,10,20,50,100]
        
        for cut in cut_off:
            result.extend(get_score(data,cut))
    else:
        result.extend(get_score(data,cut_off))
    
    return result

if __name__ == "__main__":
    main()

