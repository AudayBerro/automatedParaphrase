from translator import my_memory_translator as memory
# from translator import yandex_translator as yandex
from translator import deepl_translator as deepl
from translator import marian_translator as marian
from pos import pos_extraction as pos
from filtering import bert_filter as bert
from filtering import use_filter as use
from synonym import nltk_wordnet as nlt
from synonym import parpahraser as para
import os
import configparser
#import spacy
from datetime import datetime as dt
import argparse
import re,string
from evaluation import bleu_score,gleu_score,chrf_score,diversity_metrics

### interpolation import
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.chrf_score import sentence_chrf
from Transformers import t5_paraphrases_generation as t5

#time and color in console
import time
import datetime
#model_name='en_use_lg'

cache = {} #global variable containing single instance 

def load_library(*args):
    """ 
    Load dependencies library respecting the Singleton design pattern to avoid repetitive reload
    :param args: model to load dependencies for
    :return an instance of the model
    """
    global cache
    if args[0]=='load_spacy_nlp':# load spaCy NLP tagger model
        if not(args[0] in cache):
            cache[args[0]] =  nlt.load_spacy_nlp(args[1])
            return cache[args[0]]

    if args[0]=='load_t5':# load Huggingface T5 transformer
        if not(args[0] in cache):
            # check if seed is set
            if len(args) == 4:
                cache[args[0]] =  t5.initialisation(args[1],args[2],args[3])#args[1]=model_name; args[2]=tokenizer_name; args[3]=seed integer for reproducibility (optional)
            else:
                cache[args[0]] =  t5.initialisation(args[1],args[2])
            return cache[args[0]]#model,tokenizer,device

    if args[0]=='load_marian':# load Huggingface Marian Machine Translation Model
        if not(args[0] in cache):
            cache[args[0]] =  marian.load_model()
            return cache[args[0]]
    
    if args[0]=='load_use':
        if not(args[0] in cache):
            #args[1] = moddel name to load
            cache[args[0]] = use.load_model(args[1])
            return cache[args[0]]
    return cache[args[0]]


# load_library('load_spacy_nlp','tr','test1')

def pr_green(msg):
    """ Pring msg in green color font"""
    print("\033[92m{}\033[00m" .format(msg))

def pr_red(msg): 
    """ Pring msg in Red color font"""
    print("\033[91m {}\033[00m" .format(msg)) 

def normalize_text(text):
    """
    Remove punctuation except in real value or date(e.g. 2.5, 25/10/2015),line break and lowercase all words
    :param text: sentence to normalize
    :return return a preprocessed sentence e.g. "This is a ? 12\3  ?? 5.5 covid-19 ! ! *  & $ % ^" => "this is a 12\3 5.5 covid-19"
    """
    regex = "(?<!\w)[!\"#$%&'()*-+/:;<=>?@[\]^_`{|}~](?!\w)"

    #remove punctuation
    result = re.sub(regex, "", text, 0)

    #trim to remove excessive whitespace
    result = re.sub(' +', ' ',(result.replace('\n',' '))).strip().lower()

    return result

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

def write_to_folder(data,message,file_name):
    """
    Conserve data as file in result folder
    :param data: python dictionary containing the generated paraphrases
    :param message: a short message that describe the element to be listed
    :param file_name: file name
    """
    f = open("./result/"+file_name, "a")
    f.write(message+'\n\t'+str(data)+'\n')
    f.close()

def merge_data(dataset1,dataset2):
    """
    Merge dataset1 with dataset2
    :param dataset1: python dictionary
    :param dataset2: python dictionary
    :return a Python dictionary, Key is the initial expression and value is a list of paraphrases
    """
    for (k,v), (k2,v2) in zip(dataset1.items(), dataset2.items()):
        v.add(normalize_text(k2)) # add key of dataset2 to dataset1 list of paraphrases
        v.update(v2) # add dataset2 paraphrases list to dataset1 paraphrases list
    return dataset1

def sort_collection(pool):
    """
    This function sort the filtred BERT dictionary in descending order according to the second element of the value tuple wich is the BERT embeddong cosine similarity score
    :param pool: python dictionary, key is the initial utterance and value is a list of tuples. Tuples(paraphrase, BERT embeddong cosine similarity score)
    :return ordred Python dictionary
    """
    for key in pool:
        pool[key].sort(key = lambda x: x[1],reverse = True)
    
    return pool

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

def sbss_weak_supervision_generation(sentence,spacy_nlp):
    """
    Generate parpahrases using nltk_wordnet.py module
    :param sentence: string to generate parpahrases for
    :param spacy_nlp: spacy Universal sentence embedding model
    :return a list of 3 paraphrases generated using the SBSS part of the weak-supervision component of the pipeline
    """
    result = []
    # Generate data by Replacing only word with VERB pos-tags by synonym
    spacy_tags = ['VERB'] #list of tag to extract from sentence using spacy
    wm_tags = ['v'] #wordnet select only lemmas which pos-taggs is in wm_tags
    data1 = nlt.gui_main(sentence,spacy_tags,wm_tags,spacy_nlp,pos)
    result.append(data1)

    # Generate data by Replacing only word with NOUN pos-tags by synonym 
    spacy_tags = ['NOUN'] #list of tag to extract from sentence using spacy
    wm_tags = ['n'] #wordnet select only lemmas which pos-taggs is in wm_tags
    data2 = nlt.gui_main(sentence,spacy_tags,wm_tags,spacy_nlp,pos)
    result.append(data2)
    
    # Generate data by Replacing only word with NOUN and VERB pos-tags by synonym
    spacy_tags = ['VERB','NOUN'] #list of tag to extract from sentence using spacy
    wm_tags = ['v','n'] #wordnet select only lemmas which pos-taggs is in wm_tags
    data3 = nlt.gui_main(sentence,spacy_tags,wm_tags,spacy_nlp,pos)
    result.append(data3)

    return result

####  GRAPHICAL USER INTERFACE MODE CODE  ####

def gui_sbss(sent,spacy_nlp,flag):
    """
    Apply Weak Supervision to generate parpahrases using nltk_wordnet.py module, use this function for GUI
    :param sent: :param data: Python dictionary, key:initial utterance, value: list of paraphrases
    :param spacy_nlp: spacy Universal sentence embedding model
    :param flag: integer, flag=0 mean the pipeline start with weak-supervision, otherwise flag=1 
    :return a Python dictionary, Key:initial expression, value: list of paraphrases
    """
    result = dict()
    if flag == 0:#the pipeline start with the weak supervision SBSS component
        for k,v in sent.items():
            paraphrases = set(sbss_weak_supervision_generation(k,spacy_nlp))# convert to set to remove redundancy before adding candidate
            result[k] = list(paraphrases) #convert to list before the insertion

    elif flag == 1:#the pipeline have started with another component(e.g. Pivot-translation, T5, etc)
        for k,v in sent.items():
            candidates = set()#will contain the generated paraphrases

            #generate paraphrases for the initial expression k
            paraphrases = sbss_weak_supervision_generation(k,spacy_nlp)
            candidates.update(paraphrases)

            #generate paraphrases for each element in the values list
            if v:#check if v not empty
                for element in v:
                    paraphrases = sbss_weak_supervision_generation(element,spacy_nlp)
                    candidates.update(paraphrases)
                
                candidates.update(v)#add K list of parpahrases to result to avoid loosing previous parpahrases 
            result[k] = list(candidates)

    return result

def gui_srss_weak_supervision_generation(sent):
    """
    Apply Weak Supervision to generate data using paraphraser.py module (SRSS component)
    :param sent: python dictionary, key:initial sentence, value list of paraphrases candidates
    :return a python dictionary containing a list generated paraphrases
    """
    result = dict()
    for k,v in sent.items():
        candidates = set()

        #generate parpahrases for the initial expression k
        paraphrases = para.gui_main(k)
        candidates.update(paraphrases)

        #generate paraphrases for each element in the values list
        if v:#check if v not empty
            for element in v:
                paraphrases = para.gui_main(element)
                candidates.update(paraphrases)
            
            candidates.update(v)
        result[k] = list(candidates)

    return result

def gui_pivot_translation(sent,pivot_level,flag):
    """
    Generate Paraphrases using Pretrained Translation Model e.g. Huggingface MarianMT
    :param sent: python dictionary, key:initial sentence, value list of paraphrases candidates
    :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
    :param flag: integer, flag=0 mean the pipeline start with pivot-translation, otherwise flag=1 
    :return a Python dictionary, Key is the initial expression and value is a list of paraphrases candidates
    """
    result = dict()
    #load all the supported model
    model_list = load_library('load_marian') #for now we only support HuggingFace Marian MT and OpenNMT

    if flag == 0:
        for k,v in sent.items():
            result[k] =  marian.multi_translate(k,model_list,pivot_level)
            
    elif flag == 1:#the pipeline have started with another component(e.g. Weak-supervision, T5, etc)
        for k,v in sent.items():
            candidates = set()#will contain the generated paraphrases

            #generate paraphrases for the initial expression k
            paraphrases = marian.multi_translate(k,model_list,pivot_level)
            candidates.update(paraphrases)

            #generate paraphrases for each element in the values list
            if v:#check if v not empty
                for element in v:
                    paraphrases = marian.multi_translate(element,model_list,pivot_level)
                    candidates.update(paraphrases)
                
                candidates.update(v)#add K list of parpahrases to result to avoid loosing previous parpahrases 
            result[k] = list(candidates)

    return result

####  COMMANDE LINE MODE CODE  ####

def weak_supervision_generation2(file_path):
    """
    Apply Weak Supervision to generate data using paraphraser.py module (SRSS component)
    :param file_path: file path to folder containing initial utterances
    :return a dictionary, key initial utterance, value set of parpahrases generated using the parpahraser.py module
    """
    return para.main(file_path)

def weak_supervision_generation(file_path):
    """
    Apply Weak Supervision to generate data using nltk_wordnet.py module (SBSS component)
    :param file_path: file path to folder containing initial utterances
    :return list of parpahrases, for each sentence it return 3 paraphrases one paraphrase in each dataset(data1 replace NOUN, data2 replace VERB, data3 replace NOUN and VERB)
    """

    # Generate data by Replacing only word with VERB pos-tags by synonym
    spacy_tags = ['VERB'] #list of tag to extract from sentence using spacy
    wm_tags = ['v'] #wordnet select only lemmas which pos-taggs is in wm_tags
    data1 = nlt.main(file_path,spacy_tags,wm_tags)

    # Generate data by Replacing only word with NOUN pos-tags by synonym 
    spacy_tags = ['NOUN'] #list of tag to extract from sentence using spacy
    wm_tags = ['n'] #wordnet select only lemmas which pos-taggs is in wm_tags
    data2 = nlt.main(file_path,spacy_tags,wm_tags)
    
    # Generate data by Replacing only word with NOUN and VERB pos-tags by synonym
    spacy_tags = ['VERB','NOUN'] #list of tag to extract from sentence using spacy
    wm_tags = ['v','n'] #wordnet select only lemmas which pos-taggs is in wm_tags
    data3 = nlt.main(file_path,spacy_tags,wm_tags)

    return data1,data2,data3


def online_transaltion(file_path,api_key,valid_mail,pivot_level,cut_off):
    """
    Generate Paraphrases Using online Translator Engine e.g. Google, Yandex
    :param file_path: file path to folder containing initial utterances
    :param api_key: Online Translator API key
    :param valid_mail: valid email address to reach a translation rate of 10000 words/day in MyMemory API.
    :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return a Python dictionary, Key is the initial expression and value is a list of paraphrases
    """
    #wordnet
    print("Start weak supervision data generation ",end="")
    t = time.time()

    data1,data2,data3 = weak_supervision_generation(file_path)
    data4 = weak_supervision_generation2(file_path) #pool = nlsp.main(file_path)

    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))

    print("Start translation ",end="")
    t = time.time()
    # generate paraphrases with MyMemory API
    word_counter = 0 # 
    memory_result1,word_counter = memory.translate_list(data1,valid_mail,word_counter) #generate paraphrases through pivot-translation of data1
    memory_result2,word_counter = memory.translate_list(data2,valid_mail,word_counter) #generate paraphrases through pivot-translation of data2
    memory_result3,word_counter = memory.translate_list(data3,valid_mail,word_counter) #generate paraphrases through pivot-translation of data3
    
    result,word_counter = memory.translate_file(file_path,valid_mail,word_counter) #generate paraphrases through pivot-translation of initial utterances folder

    # merge memory_result1, memory_result2, memory_result3 with result
    result= merge_data(result,memory_result1)
    result= merge_data(result,memory_result2)
    result= merge_data(result,memory_result3)

    # generate paraphrases with Yandex Translator API
    # yandex_result1 = yandex.translate_list(data1,api_key,pivot_level)
    # yandex_result2 = yandex.translate_list(data2,api_key,pivot_level)
    # yandex_result3 = yandex.translate_list(data3,api_key,pivot_level)

    # generate paraphrases with DeepL API
    deepl_result1 = deepl.translate_list(data1,api_key,pivot_level)
    deepl_result2 = deepl.translate_list(data2,api_key,pivot_level)
    deepl_result3 = deepl.translate_list(data3,api_key,pivot_level)

    # merge memory_result1, memory_result2, memory_result3 with result
    result= merge_data(result,deepl_result1)
    result= merge_data(result,deepl_result2)
    result= merge_data(result,deepl_result3)

    # yandex_result = yandex.translate_file(file_path,yandex_api_key,pivot_level)
    deepl_result = deepl.translate_file(file_path,api_key,pivot_level)
    extracted_pos = pos.pos_extraction(file_path)
    # yandex_paraphrases = yandex.translate_dict(extracted_pos,yandex_api_key,pivot_level)
    deepl_paraphrases =  deepl.translate_dict(extracted_pos,api_key,pivot_level)

    # merge all dictionary into one
    for key,values in result.items():
        values.update(deepl_result[key])
        values.update(deepl_paraphrases[key])
        result[key] = values
    
    result = merge_data(result,data4)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))

    write_to_folder(result,"Generated Paraphrases:","paraphrases.txt")
    #universal sentence encoder filtering
    print("Start Universal Sentence Encoder filtering ",end="")
    t = time.time()
    use_filtered_paraphrases = use.get_embedding(result)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
    # write_to_folder(use_filtered_paraphrases,"Universal Sentence Encoder Filtering:","paraphrases.txt")

    print("Start BERT filtering ",end="")
    t = time.time()
    bert_filtered_paraphrases = bert.bert_selection(use_filtered_paraphrases)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
    # write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")

    # sort the dictionary
    bert_filtered_paraphrases = sort_collection(bert_filtered_paraphrases)
    
    if cut_off > 0:
        print("Start cut-off ",end="")
        final_result = apply_cut_off(bert_filtered_paraphrases,cut_off)
        print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
        write_to_folder(final_result,"Final Paraphrases List:","paraphrases.txt")
    else:
        write_to_folder(bert_filtered_paraphrases,"Final Paraphrases List:","paraphrases.txt")
    
    return bert_filtered_paraphrases

def pretrained_transaltion(file_path,pivot_level,cut_off):
    """
    Generate Paraphrases using Pretrained Translation Model e.g. Huggingface MarianMT
    :param file_path: file path to folder containing initial utterances
    :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
    :param cut_off: integer that indicate how many parpahrases to select, e.g. cut_off = 3 will only select top highest 3 semantically related parpahrases and drop the rest
    :return a Python dictionary, Key is the initial expression and value is a list of paraphrases
    """
    #load all the model
    # print("load model")
    model_list = marian.load_model()
    
    #wordnet
    print("Start weak supervision data generation ",end="")
    t = time.time()

    data1,data2,data3 = weak_supervision_generation(file_path)
    data4 = weak_supervision_generation2(file_path) #pool = nlsp.main(file_path)

    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))

    print("Start translation ", end="")
    t = time.time()
    # generate paraphrases with Marian MT
    result1 = marian.translate_list(data1,model_list,pivot_level)
    result2 = marian.translate_list(data2,model_list,pivot_level)
    result3 = marian.translate_list(data3,model_list,pivot_level)
    
    result = marian.translate_file(file_path,model_list,pivot_level) #  (file_path,model_list,pivot_level)

    # merge result1, result2, result3 with result
    result= merge_data(result,result1)
    result= merge_data(result,result2)
    result= merge_data(result,result3)

    result = merge_data(result,data4)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))

    write_to_folder(result,"Generated Paraphrases:","paraphrases.txt")
    #universal sentence encoder filtering
    print("Start Universal Sentence Encoder filtering ", end="")
    t = time.time()
    use_filtered_paraphrases = use.get_embedding(result)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
    # write_to_folder(use_filtered_paraphrases,"Universal Sentence Encoder Filtering:","paraphrases.txt")
    
    print("Start BERT filtering ", end="")
    bert_filtered_paraphrases = bert.bert_selection(use_filtered_paraphrases)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
    # write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")

    # sort the dictionary
    bert_filtered_paraphrases = sort_collection(bert_filtered_paraphrases)
    
    if cut_off > 0:
        print("Start cut-off ", end="")
        t = time.time()
        final_result = apply_cut_off(bert_filtered_paraphrases,cut_off)
        print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
        write_to_folder(final_result,"Final Paraphrases List:","paraphrases.txt")
    else:
        write_to_folder(bert_filtered_paraphrases,"Final Paraphrases List:","paraphrases.txt")
    
    return bert_filtered_paraphrases

def main():
    # required arg
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True) # -f data set file name argument
    parser.add_argument('-g') # if -g is defined use google_translator.translate method not translate_wrapper
    parser.add_argument('-l') # -l integer that indicate the pivot language level, single-pivot or multi-pivot range between 0 and 2
    parser.add_argument('-p') # use pretrained translator(p==true - MarianMT) or online translator engine(p==false - Yandex,Google Translator)
    parser.add_argument('-c') # cut-off criteria to stop paraphrasing, default c=0 which mean don't apply cut-off
    args = parser.parse_args()
    
    # load configs from config.ini file
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(os.path.join(os.path.dirname(__file__), ".","config.ini"))
    my_memory_config = config["MYMEMORY"]
    yandex_config = config["YANDEX"]
    google_config = config["GOOGLE"]
    deepl_config = config['DEEPL']

    try:
        if str(args.p) == "None":#if -p not defined set default value to true
            args.p="true"
        if args.p == "false":
            if "email" not in my_memory_config or my_memory_config["email"] == "":
                raise Exception("Define a Valid email address for MyMemory API in config.ini")
            else:
                valid_mail = my_memory_config['email']
            if "api_key" not in yandex_config or yandex_config["api_key"] == "":
                raise Exception("Yandex Translate API token is not defined in config.ini")
            else:
                yandex_api_key = yandex_config["api_key"]
            if "api_key" not in deepl_config or deepl_config["api_key"] == "":
                raise Exception("DeepL API Authentication Key not defined in config.ini")
            else:
                deepl_api_key = deepl_config["api_key"]
            if args.g:#flag g specify to use Official Google Traslator API not a wrapper
                if "api_key" not in google_config or google_config["api_key"] == "":
                    raise Exception("Google Translate API token is not defined in config.ini")
                else:
                    google_api_key = google_config['api_key']
        if args.l:
            pivot_level = int(args.l)
            if pivot_level<0 or pivot_level>2:
                raise Exception("Pivot-level value should be 0,1 or 2")
        else:
            pivot_level = 0
        
        if args.c:
            cut_off = int(args.c)
            if cut_off<0:
                raise Exception("Cut-off parameter value should be greater or equal to 0")
        else:
            cut_off = 0 # default value

    except Exception as e:
        print(str(e))
        exit()

    file_path = os.path.join(os.path.dirname(__file__), ".", "dataset/"+args.f) # data to paraphrase

    t1 = time.time() # to compute overall time execution
    now = dt.now()
    start_time = now.strftime("%H:%M:%S")
    pr_green("Starting time: "+start_time)

    if args.p=="true":
        paraphrases = pretrained_transaltion(file_path,pivot_level,cut_off)
    else:
        paraphrases = online_transaltion(file_path,deepl_api_key,valid_mail,pivot_level,cut_off)
    

    # compute diversity metrics
    print("\nCompute Mean-TTR, Mean-PINC and DIV scores: ")
    diversity_score = diversity_metrics.main(paraphrases,cut_off)

    for k,v in diversity_score.items():
        print("\t============================================================")
        print("\t                  Cut_off parpameter = ",k,"            ")
        print("\t============================================================")
        print("\t\tMean TTR: ", v[0]["Mean TTR"])
        print("\t\tMean PINC: ", v[1]["Mean PINC"])
        print("\t\tDiversity: ", v[2]['Diversity'])
    
    paraphrases = remove_cosine_score(paraphrases)

    # compute BLEU-Score of generated paraphrases
    print("\nCompute BLEU, GLEU and CHRF scores: ")
    bleu_score.main(paraphrases,cut_off)
    gleu_score.main(paraphrases,cut_off)
    chrf_score.main(paraphrases,cut_off)
    
    t2 = "Overall elapsed time: "+str(datetime.timedelta(0,time.time()-t1))
    pr_green(t2)

def generate_from_gui(sentence,pipeline_config,pivot_level=None,pre_trained=None):
    """
    Generate parpahrases using Graphical User Interface(GUI) of the pipeline implemented in index.html 
    :param sentence: user sentence to parpahrase obtained from the GUI. Value from templates/index.html <input type="text" name="user_utterance"/>
    :param pipeline_config: user configuration of the pipline from the GUI. Value from templates/index.html <select id="monselect" name="configuration">
    :param pivot_level: pivot translation level to use for the Pivot-Translation component. Value from templates/index.html <input type="radio" name="pivot_level"/>
    :param pre_trained: Machine Translation model option to use for the Pivot-Translation component. Value from templates/index.html <input type="radio" name="pre_trained_mt"/>
    :return a Python dictionary, key:initial expression, value: list of paraphrases
    """
    ####################### (1) OVER-GENERATION STAGE ##########################
    #                            _..----------.._                       
    #                   .-=""        _       ""=-.                  
    #                .-"    _.--""j _\""""--._    "-.               
    #             .-"  .-i   \   / / \;       ""--.  "-.            
    #           .'  .-"  : ( "  : :                "-.  `.          
    #         .'  .'      `.`.   \ \                  `.  `.        
    #        /  .'      .---" ""--`."-./'---.           `.  \       
    #       /  /      .'                    '-.           \  \      
    #      /  /      /                         `.          \  \     
    #     /  /      /                  ,--._   (            \  \    
    #    ,  /    '-')                  `---'    `.           \  .   
    #   .  :      .'                              "-._.-.     ;  ,  
    #   ;  ;     /            :;         ,-"-.    ,--.   )    :  :  
    #  :  :     :             ::        :_    "-. '-'   `,     ;  ; 
    #  |  |     :              \\     .--."-.    `._ _   ;     |  | 
    #  ;  ;     :              / "---"    "-."-.    l.`./      :  : 
    # :  :      ;             :              `. "-._; \         ;  ;
    # ;  ;      ;             ;                `..___/\\        :  :
    # ;  ;      ;             :                        \\    _  :  :
    # :  :     /              '.                        ;;.__)) ;  ;
    #  ;  ; .-'                 "-...______...--._      ::`--' :  : 
    #  |  |  `--'\                                "-.    \`._, |  | 
    #  :  :       \                                  `.   "-"  ;  ; 
    #   ;  ;       `.                                  \      :   ' 
    #   '  :        ;                                   ;     ;  '  
    #    '  \    _  : :`.                               :    /  /   
    #     \  \   \`-' ; ; ._                             ;  /  /    
    #      \  \   `--'  : ; "-.                          : /  /     
    #       \  \        ;/     \                         ;/  /      
    #        \  `.              ;                        '  /       
    #         `.  "-.   bug    /                          .'        
    #           `.   "-..__..-"                         .'          
    #             "-.                                .-"            
    #                "-._                        _.-"               
    #                    """---...______...---"""
    #
    #   Art by Blazej Kozlowski - https://www.asciiart.eu
    ############################################################################

    # initialise flag
    flag = 0

    # T5 pre-trained paraphraser model to load
    model_name="auday/paraphraser_model2"
    num_seq = 40 # default 10
    max_len = 256
    #t5_paraphraser(sent,model,tokenizer,device,flag=0,num_seq=40,max_len=256): initialisation(model_name="auday/paraphraser_model2",tokenizer='t5-base',seed=None)
    

    #convert sentence to dictionary
    sentence = {sentence:[]}
    # pipeline configuration
    if pipeline_config == "c1":# Pivot-Translation
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(sentence,pivot_level,flag)

    elif pipeline_config == "c2":# Weak-supervision
        #start the pipeline with Weak-Supervision SBSS component
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(sentence,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

    elif pipeline_config == "c3":# T5
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

    elif pipeline_config == "c4":# Weak-Supervision => Pivot-Translation
        ################################################
        ### Start the pipeline with Weak-Supervision ###
        ################################################
        #start the pipeline with Weak-Supervision SBSS component
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(sentence,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

        #############################
        ### Run Pivot-Translation ###
        #############################
        flag = 1 # set flag to 1
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(result,pivot_level,flag)

    elif pipeline_config == "c5":# Weak-Supervision => T5
        ################################################
        ### Start the pipeline with Weak-Supervision ###
        ################################################
        #start the pipeline with Weak-Supervision SBSS component
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(sentence,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

        #############################
        ### Run T5 ###
        #############################
        flag = 1 # set flag to 1
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

    elif pipeline_config == "c6":# Weak-Supervision => Pivot-Translation => T5
        ################################################
        ### Start the pipeline with Weak-Supervision ###
        ################################################
        #start the pipeline with Weak-Supervision SBSS component
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(sentence,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

        #############################
        ### Run Pivot-Translation ###
        #############################
        flag = 1 # set flag to 1
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(result,pivot_level,flag)

        #############################
        ### Run T5 ###
        #############################
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

    elif pipeline_config == "c7":# Weak-Supervision  => T5 => Pivot-Translation
        ################################################
        ### Start the pipeline with Weak-Supervision ###
        ################################################
        #start the pipeline with Weak-Supervision SBSS component
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(sentence,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

        #############################
        ### Run T5 ###
        #############################
        flag = 1
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

        #############################
        ### Run Pivot-Translation ###
        #############################
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(result,pivot_level,flag)

    elif pipeline_config == "c8":# Pivot-Translation => Weak-Supervision
        #################################################
        ### Start the pipeline with Pivot-Translation ###
        #################################################
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(sentence,pivot_level,flag)

        #############################
        ### Run Weak-Supervision ###
        #############################
        flag = 1 # set flag to 1
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(result,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

    elif pipeline_config == "c9":# Pivot-Translation => T5
        #################################################
        ### Start the pipeline with Pivot-Translation ###
        #################################################
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(sentence,pivot_level,flag) 

        #############################
        ### Run T5 ###
        #############################
        flag = 1
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

    elif pipeline_config == "c10":# Pivot-Translation => Weak-Supervision => T5
        #################################################
        ### Start the pipeline with Pivot-Translation ###
        #################################################
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(sentence,pivot_level,flag)

        #############################
        ### Run Weak-Supervision ###
        #############################
        flag = 1 # set flag to 1
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')
        result = gui_sbss(result,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

        ##############
        ### Run T5 ###
        ##############
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

    elif pipeline_config == "c11":# Pivot-Translation => T5 => Weak-Supervision
        #################################################
        ### Start the pipeline with Pivot-Translation ###
        #################################################
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(sentence,pivot_level,flag)

        #############################
        ### Run T5 ###
        #############################
        flag = 1 # set flag to 1
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

        #############################
        ### Run Weak-Supervision ###
        #############################
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(result,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

    elif pipeline_config == "c12":# T5 => Weak-Supervision
        ##################################
        ### Start the pipeline with T5 ###
        ##################################
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

        #############################
        ### Run Weak-Supervision ###
        #############################
        flag = 1 # set flag to 1
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(result,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

    elif pipeline_config == "c13":# T5 => Pivot-Translation
        ##################################
        ### Start the pipeline with T5 ###
        ##################################
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

        #############################
        ### Run Pivot-Translation ###
        #############################
        flag = 1 # set flag to 1
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(result,pivot_level,flag)

    elif pipeline_config == "c14":# T5 => Pivot-Translation => Weak-Supervision
        ##################################
        ### Start the pipeline with T5 ###
        ##################################
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

        #############################
        ### Run Pivot-Translation ###
        #############################
        flag = 1 # set flag to 1
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(result,pivot_level,flag)

        #############################
        ### Run Weak-Supervision ###
        #############################
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(result,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)
    elif pipeline_config == "c15":# T5 => Weak-Supervision => Pivot-Translation
        ##################################
        ### Start the pipeline with T5 ###
        ##################################
        # load T5 model and tokenizer
        t5_model = load_library('load_t5',model_name,'t5-base')#t5_model[0]=model; t5_model[1]=tokenizer; t5_model[2]=device
        result = t5.t5_paraphraser(sentence,t5_model[0],t5_model[1],t5_model[2],flag,num_seq,max_len)

        #############################
        ### Run Weak-Supervision ###
        #############################
        flag = 1 # set flag to 1
        # load spaCy USE embedding model
        spacy_nlp = load_library('load_spacy_nlp','en_use_lg')

        result = gui_sbss(result,spacy_nlp,flag)

        #Run Weak-Supervision SRSS component
        result = gui_srss_weak_supervision_generation(result)

        #############################
        ### Run Pivot-Translation ###
        #############################
        #convert pivot_level to integer
        pivot_level = int(pivot_level)
        #run pivot translation component
        result = gui_pivot_translation(result,pivot_level,flag)
    else:
        result = {"Error":"Error in the pipeline configuration"}
    
    ############# (2) CANDIDATE SELECTION STAGE ###################
    #                                  ,        ,
    #                                 /(        )`
    #                                 \ \___   / |
    #                                 /- _  `-/  '
    #                                (/\/ \ \   /\
    #                                / /   | `    \
    #                                O O   ) /    |
    #                                `-^--'`<     '
    #                               (_.)  _  )   /
    #                                `.___/`    /
    #                                  `-----' /
    #                     <----.     __ / __   \
    #                     <----|====O)))==) \) /====
    #                     <----'    `--' `.__,' \
    #                                  |        |
    #                                   \       /
    #                              ______( (_  / \______
    #                            ,'  ,-----'   |        \
    #                            `--{__________)        \/
    #
    ################################################################
    #  by: Kevin Woods - https://www.asciiart.eu/computers/linux   #
    ################################################################
    return result