#this code crawl the WebQuestion datasets github repo
import requests
import json
import urllib
import unidecode # convert unicode string to ascii string
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def get_bleu_score(dataset):
    """
    This function return the dataset BLEU score
    :param dataset: the data to calculate the BLUE_Score
    :return bleu score and the number of utterance
    """
    cc = SmoothingFunction()
    bleu = sentence_bleu
    def get_smooth(hyp,ref):
                #print("\t"+str(hyp)+": ",sep="")
                ##a = bleu(ref, hyp,weights=(0.5, 0.5, 0, 0), smoothing_function=cc.method1)
                #a = bleu(ref, hyp, smoothing_function=cc.method1)
                #print("\tm1= "+str(a),sep="")

                #a = bleu(ref, hyp, smoothing_function=cc.method2)
                #print("\tm2= "+str(a),sep="") # this is the best smoothing method view desktop/smoothing_method_experiment.txt 

                #a = bleu(ref, hyp, smoothing_function=cc.method3)
                #print("\tm3= "+str(a),sep="")

                #b = bleu(ref, hyp, smoothing_function=cc.method4)
                #print("\tm4= "+str(b),sep="")
                #print()
                return bleu(ref, hyp, smoothing_function=cc.method2)
    
    counter = 0
    data_set_bleu_score = 0 #bleu score of all dataset
    for k,v in dataset.items():
        counter +=1
        reference = [k.lower().split(" ")]
        utterance_bleu_score = 0 # current utterance bleu_Score = average_paraphrase_bleu_score
        print(k)
        for cand in v:
            candidate = cand[0].lower().split(" ")
            #get_smooth(candidate,reference)
            #parpahrase_bleu_Score = sentence_bleu(reference,candidate,weights=(1,0,0,0))
            parpahrase_bleu_Score = get_smooth(candidate,reference)
            print("\tcandidate bleu score: "+cand[0]+"= "+str(parpahrase_bleu_Score),sep="")
            utterance_bleu_score += parpahrase_bleu_Score
            print("\t - average= "+str(utterance_bleu_score))
        
        if utterance_bleu_score > 0:
            utterance_bleu_score = utterance_bleu_score / len(v)
            print("\t\tBLEU("+k+")= "+str(utterance_bleu_score))
        data_set_bleu_score += utterance_bleu_score
    bleu = data_set_bleu_score / len(dataset)
    print("================================================\n\tBLEU = "+str(bleu)+"\n\tUtterance number: "+str(counter)+"\n================================================")
    return bleu,counter

url = "https://raw.githubusercontent.com/ysu1989/GraphQuestions/master/freebase13/graphquestions.testing.json"
req = requests.get(url)
print(req)
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

# save result in csv file
f = open("./dataset/web_Question_crawled_dataset.csv","a")

f.write("question\tnumber of paraphrases\tparaphrases\n")
for k,v in result.items():
    f.write(k+"\t"+str(len(v)))
    for e in v:
        f.write("\t"+e)
    f.write("\n")

f.close()