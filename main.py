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
import time
import datetime
import argparse
import re,string
from evaluation import bleu_score,gleu_score,chrf_score,diversity_metrics

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


def weak_supervision_generation2(file_path):
    """
    Apply Weak Supervision to generate data using paraphraser.py module
    :param file_path: file path to folder containing initial utterances
    :return a dictionary, key initial utterance, value set of parpahrases generated using the parpahraser.py module
    """
    return para.main(file_path)

def weak_supervision_generation(file_path):
    """
    Apply Weak Supervision to generate data using nltk_wordnet.py module
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
    # generate paraphrases with MyMemory API
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
    
    # compute BLEU-Score of generated paraphrases
    print("Compute BLEU, GLEU and CHRF scores: ")
    bleu_score.main(paraphrases,cut_off)
    gleu_score.main(paraphrases,cut_off)
    chrf_score.main(paraphrases,cut_off)
    print("Overall elapsed time: ",str(datetime.timedelta(0,time.time()-t1)))

def test_model():
    data = {'what are the violent events that started on 1995-04-07?': {'what constitute the violent event that commence on 1995 - 04 - 07', 'what is the incident of violence that began on 1995-04-07?', 'what violent events occurred between 1995 and 04 and 07?', 'what was the incident of violence that took place on 4 july 1995?', 'events that have already occurred', 'what violent events have started', 'what constitute the violent events that start on 1995 - 04 - 07', 'what constitute the violent events that commence on 1995 - 04 - 07', 'what constitute the violent events that start up on 1995 - 04 - 07', 'what constitute the violent upshot that started on 1995 - 04 - 07', 'what violent events were triggered?', 'what are the violent events that start up on 1995 - 04 - 07', 'what are the violent upshot that commence on 1995 - 04 - 07', 'what constitute the violent upshot that originate on 1995 - 04 - 07', 'what be the violent upshot that initiate on 1995 - 04 - 07', 'what are the violent events that have their origin in the years 1995 - 04 - 07?', 'what are the violent events that originate on 1995 - 04 - 07', 'what are the violent events that began in 1995-04-07?', 'which violent events have been triggered?', 'what violent incidents occurred between 1995-04 and 2007?'}, 'on the date of 1882-04-26 what athletes were born?': {'on the appointment of 1882 - 04 - 26 what athlete be gestate', 'on the date of 1882 - 04 - 26 what athletes be gestate', 'on the date of 1882 - 04 - 26 what athlete were born', 'which players were held in the years 1882 - 26.04.', 'which athletes were born on date 1882-04-26?', 'date of birth of the athletes', 'on day 1882-04-26, which athletes were born?', 'on 1882-04-26, which athletes were born?', 'on the appointment of 1882 - 04 - 26 what athletes be birth', 'in 1882 - 04 - 26, which athlete was born?', 'on the date of 1882 - 04 - 26 what athlete were gestate', 'on the date of 1882 - 04 - 26 which athlete was born?', 'exactly 1882-04-26, which player was held?', 'at exactly 1882 - 04 - 26, which athlete was detained?', 'which players were born on april 26, 1882?', 'the day the athlete was born.', 'on the date of 1882-04-26 what athletes were born?', 'on the date 1882 - 04 - 26, which athlete was born?', 'date of birth athletes', 'on the date 1882 - 04 - 26, which athletes were retained?', 'on the date of 1882 - 04 - 26 what athlete be born', 'as of 1882-04-26 which athletes were detained?', 'on the date of 1882 - 04 - 26 what athletes be birth', 'until the date of 1882 - 04 - 26, which players were born?', "athletes' date of birth", 'which athletes were on date 1882 - 04 - 26 in prison?', 'which athletes were in prison on the date 1882 - 04 - 26?', 'which players were brought to a halt on 26 april 1882?'}, 'by whom was paul the apostle influenced?': {'paul the apostle', 'who was influenced by the apostle paul?', 'the influences of who are.', 'by whom was paul the apostle influence', 'by whom be paul the apostle influenced', 'by whom was paul the apostle?', 'who was the effect of the apostle paul?', 'who are the influences.', 'paul, the apostle', 'by whom constitute paul the apostle influenced', '"and by whom was he arrested?', 'by whom was paul the apostle influenced?', 'saint malo', 'by whom be paul the apostle influence', 'influenced by the apostle', 'who coined the phrase? who used it?', 'saint pauls', 'by whom was the apostle paul influenced?', 'who influenced the apostle paul?', 'who was influenced by the apostle', 'who was the apostle paul?', 'who was influenced by apostle paul?', 'whose influences are.', 'impact of apostles', 'who was affected by the apostle paul?', 'who the influences are.', 'who was the apostle paul influenced?', 'who was the influence of the apostle paul?', 'paul, apostle of the resurrection', 'by whom did the apostle paul been influenced?', "impact of the apostles'", 'by whom constitute paul the apostle influence', 'by whom was paul the apostle influenced', 'saint paul', 'through whom was the influence of the apostle paul?'}, 'naruto is published by which company?': {'naruto is published which company?', 'which company does naruto publish?', 'not released by the bus company', "naruto is publish by which ship's company", 'naruto publishes what company?', 'naruto publishes which company of ships?', 'which company', 'any organization', 'naruto is published by what shipping company?', 'naruto is a publishing house', 'what company is naruto serving?', 'by which shipping company is naruto published?', 'publisher naruto.', 'naruto is published by which companionship', 'naruto is the publisher.', 'which ship company does naruto publish?', 'what ship company is naruto publishing?', 'what shipping company publishes naruto?', 'what modern society!', 'naruto is published by which company?', 'naruto is published which shipping company?', 'naruto is published by what company?', 'naruto is publish by which companionship', "naruto is published by which ship's company", "naruto is published by which ship's company?", 'naruto is release by which company', 'publishing company naruto', 'what a company.', 'what company do naruto out?', 'by which naruto navigation company is published?', 'naruto publishing company', 'what company publishes naruto?', 'naruto is publish by which society', 'naruto is a publishing company', 'naruto is published by which company', 'what shipping company is naruto published?', 'naruto is published by which shipping company?', 'naruto is published by which society', 'who publishes naruto?', 'which company publishes naruto?', 'naruto is release by which society', 'naruto is issued by which shipping company?', 'naruto is publish by which company', 'naruto is the editor.'}, 'which model years of revival generation pontiac gto are available?': {'which model long time of revitalization generation pontiac gto are available', 'ancient pontiac does what example is the year of revival?', 'which modeling years of revitalisation contemporaries pontiac gto be available', 'how is the exemplary revival year available simultaneously with pontiac gto?', 'which model years of resurgence generation pontiac gto are available', 'which modelling year of revival generation pontiac gto be available', 'which modelling year of revitalisation generation pontiac gto constitute available', 'which model age of revitalization generation pontiac gto are available', 'which modeling years of revitalization generation pontiac gto constitute available', 'what is the model year of pontiac renaissance generation?', 'which model years of the pontiac gto generation of resuscitation are available?', 'which model years of revitalization contemporaries pontiac gto are available', 'which mannequin years of revitalisation generation pontiac gto be available', 'what was the model year of the pontiac renaissance generation?'}, 'what is needed to prepare cuba libre?': {'what does it take to make cuba libre?', 'what do you need to make the cuban book?', 'what do i need to make a cubulibre?', 'what is needed to make cuba libra?', 'i will make sure you have everything you need.', 'what is necessitate to ready cuba libre', 'what is demand to prepare cuba libre', 'what needs to change?', 'do whatever is necessary', 'what is needed to prepare cuba libre', 'what is required to get started?', 'what is want to ready cuba libre', "what do i need to make cuba's libra?", 'what is need?', 'what do i need to make cuba libre?', 'take steps to', 'what does this require?', 'what do you need to prepare for cuba libre?', 'what do i need to prepare for cuba libre?', 'what is want to organize cuba libre', 'what is demand to ready cuba libre', 'what is require to prepare cuba libre', 'what is necessitate to prepare cuba libre', 'what does it take to make a cuba free?', 'what is require to cook cuba libre', 'well-preparedness is the best policy', 'what does it take to prepare cuba libre?', 'what does it take to prepare for free cuba?', 'why prepare cuba libre?', 'what is need to cook cuba libre', 'recommended actions the media are encouraged to:', 'do whatever is necessary to', 'what does it take to make a cuba libre?', 'what does it take to cook cuban libre?', 'what is needed to organize cuba libre', 'what do you need to cook cuba libre?', 'what does it take to make a free cuba?', 'what is need to organize cuba libre', 'what is needed to cook cuba libre', 'what is need to ready cuba libre', 'what is needed to make cuba libre', 'what is a "need"', 'what is require to ready cuba libre', 'adequate preparation', 'what is demand to make cuba libre', 'what is a', 'what is needed?', 'what do you need to make cuba free?', 'what needs to do cuba libre?', 'what does it take to make a cuban libra?', 'what you need to prepare', 'what is needed to prepare cuba libre?', 'what does it take to make cuba libra?', 'what is needed to organise cuba libre', 'what is want to prepare cuba libre', 'what is want to cook cuba libre', 'what is necessitate to make cuba libre', 'to be prepared', 'what to eat', 'why make a cuba libra?', 'what do you need to make the cuban pound?', 'what is require to organise cuba libre', 'what is the necessity to cook cuba libre?', 'what is necessitate to organize cuba libre', 'what is need to prepare cuba libre', 'what is needed to ready cuba libre', 'i will make sure you have what you need.', 'what does it take to make a cuban libre?'}, 'which company produces pic microcontroller?': {'the company produces pic microcontrollers.', 'which company make photograph microcontroller', 'which company manufactures photographic microcontrollers?', 'microcontroller camera company', 'which companionship create pic microcontroller', 'which company produce picture microcontroller', 'microcontroller company pic', 'which society make pic microcontroller', 'which company produces the pic microcontroller?', 'the company manufactures pic microcontrollers', 'which society develop pic microcontroller', 'which companionship produces photo microcontroller', 'the company produces microcontroller images', 'microcomputer company pic', 'which company develop pic microcontroller', 'which company manufactures the pic microcontroller?', 'which shipping companies manufacture photomicrocontrollers?', "which ship's company produces photograph microcontroller", 'which company produces pic microcontroller', 'what maritime company manufactures photomicrocontroller?', 'which company manufactures the microcontroller?', "which ship's company produce photograph microcontroller", 'any organization', 'which companionship produces photograph microcontroller', 'which companionship produces pic microcontroller', 'which company produces picture microcontroller', 'company, product', 'which company acquire pic microcontroller', 'photo of the microcontroller company', 'which company create photo microcontroller', 'which company manufactures the peak microcontroller?', 'which company produces pic microcontroller?', 'which company produces photo microcontroller', 'where is the shipping company manufacturing the photo microcomputer?', 'which companionship acquire pic microcontroller', 'the company produces peak microcontrollers', 'which society produce pic microcontroller', 'which company is manufacturing pic microcomputer?', 'what products does your company produce or process?', 'the company manufactures pic microcontroller', 'which of the following does your company produce/process?', 'which shipping company manufactures photographic microcontrollers?', 'what product of service does your business provide?', 'which society develop photo microcontroller', 'what modern society!', 'which society produces photo microcontroller', 'which society create pic microcontroller', 'which ship company produces a photographic microcontroller?', 'which company produces pic microcontrollers?', 'which society get photo microcontroller', 'which companionship produces picture microcontroller', 'what shipping company produces photographic microcontrollers?', 'which companionship develop pic microcontroller', 'which company makes pic microcontrollers?', 'which shipping company manufactures photographic microcontroller?', 'which company produces the microcontroller image?', 'which company get pic microcontroller', 'which company produces a microcontroller?', 'which company manufactures photographic microcontroller?', 'which shipping company produces photographic microcontrollers?', 'which company create picture microcontroller', 'which company make pic microcontroller', 'which company create pic microcontroller', 'which company manufactures peak microcontrollers?', 'which shipping company manufactures the photomicrocontroller?', 'which company produces photographic microcontrollers?', 'which company produce pic microcontroller', 'which company makes pic microcontroller?', 'which company manufactures the microcontroller image?', 'which society produces pic microcontroller', 'which company produces photograph microcontroller', 'which company get photo microcontroller', 'which society create photograph microcontroller', 'which company'},'what sex transmitted diseases are carried by the same type of agent as tuberculosis?': {'what sex impart diseases be pack by the same type of agent as tuberculosis', 'what sex impart diseases are carry by the same type of agent as tuberculosis', 'sexually transmitted diseases are carriers of tuberculosis of the same pathogen type.', 'what sex broadcast diseases be hold by the same type of agent as tuberculosis', 'what sex transmitted diseases be carried by the same type of agent as tuberculosis', 'what sex send diseases constitute impart by the same type of agent as tuberculosis', 'what sex transmitted diseases are carried by the same type of agent as tuberculosis?', 'what type of sexual activity of the transmitted disease has the same role of agent as tuberculosis?', 'what sex transmitted diseases are take by the same type of agent as tuberculosis', 'what sex impart diseases are transport by the same type of agent as tuberculosis', 'sexually transmitted diseases are transmitted with tuberculosis, which is a pathogen of the same type.', 'what kind of sexual activity plays the same role as tuberculosis?', 'sexually transmitted diseases are carriers of tuberculosis of the same type of drug', 'what sex transmitted disease constitute carry by the same type of agent as t.b.', 'does the sexual activity of the epidemic have the same drug role as that of tuberculosis?', 'what sex broadcast diseases constitute transport by the same type of agent as tuberculosis', 'what sex broadcast disease are hold by the same type of agent as tb', 'what sex transmit disease be transport by the same type of agent as tb', 'what sex transfer diseases are carried by the same type of broker as t.b.', 'what sex transmit diseases be transport by the same type of broker as t.b.', 'which sex transmits diseases carried by the same type of agent as tuberculosis?', 'what sexual activity are transmitted diseases carried by the same character of agent role as tuberculosis?', 'which sexually transmitted diseases have a similar role to tuberculosis?', 'what sexually transmitted diseases are transmitted by the same kind of tuberculosis pathogens?', 'what stds are transmitted by the same pathogens as tuberculosis?', 'what stds have the same pathogens as tuberculosis?', 'what sexuality impart diseases be pack by the same type of agent as tb', 'which sexually transmitted diseases are transmitted by the same type of pathogen as tuberculosis?', 'what are the sexually transmitted diseases transmitted by the same type of pathogen than tuberculosis?', 'sexually transmitted diseases are transmitted together with tuberculosis, which is a pathogen of the same type.', 'what gender impart diseases are carried by the same type of broker as tb', 'what sex impart diseases are carried by the same type of agent as tuberculosis', 'what type of sexual activity plays the same role as tb?', 'what gender transmit disease be transport by the same type of agent as tb', 'what are the sexually transmitted diseases that have the same character as tuberculosis pathogens?', 'which sexually transmitted diseases are transmitted by the same type of tuberculosis pathogen?', 'what gender transmitted disease are carried by the same type of agent as tb', 'what sex send diseases be impart by the same type of agent as tuberculosis', 'what sex transfer diseases are carried by the same type of agent as tuberculosis', 'what sex impart diseases are carry by the same type of broker as t.b.', 'what sex broadcast diseases be transmit by the same type of agent as tuberculosis', 'what sex send diseases are carried by the same type of agent as tuberculosis', 'what sex transmitted diseases are carried by the same type of broker as t.b.', 'what sex broadcast diseases constitute carry by the same type of agent as tuberculosis', 'what sex impart diseases constitute transport by the same type of agent as tuberculosis', 'what sex impart diseases be carry by the same type of agent as tuberculosis', 'what sexuality impart disease constitute impart by the same type of agent as tuberculosis', 'what gender impart disease be impart by the same type of agent as tuberculosis', 'what sexuality broadcast disease constitute transport by the same type of agent as tb', 'what sexually transmitted diseases transfers the same type of tuberculosis causative agent?', 'what sex transmit diseases constitute transport by the same type of agent as tuberculosis', 'what type of sexual intercourse transmits the disease in the same way as tuberculosis?', 'what sexual activity transmits the disease, which has the same character as the tuberculosis pathogen?', 'what sex impart diseases constitute impart by the same type of agent as tuberculosis', 'what gender broadcast diseases constitute carried by the same type of broker as tuberculosis', 'what gender impart disease are carried by the same type of broker as tuberculosis', 'which sexually transmitted diseases have the same nature as tuberculosis pathogens?', 'as with tuberculosis, what kind of sexual activity mediates illness?', 'what sex transmitted diseases are carried by the same type of agent as t.b.', 'what sexual activity transmits a disease that has the same nature as a tuberculosis agent?', 'what gender transmit diseases be take by the same type of broker as tuberculosis', 'what stds are carried by the same types of pathogens as tuberculosis?', 'what sex broadcast disease constitute transport by the same type of agent as tuberculosis', 'what sex transmitted diseases are transmit by the same type of agent as tuberculosis', 'sexually transmitted diseases carry tuberculosis of the same drug type in itself', 'sexually transmitted diseases are carriers of tuberculosis of the same type of pathogen.', 'what sexually transmitted diseases carry the same type of tuberculosis pathogen?', 'what sex transmitted diseases are carry by the same type of agent as tuberculosis', 'what sex transmitted diseases are pack by the same type of broker as tuberculosis', 'what sex transmitted disease are carried by the same type of agent as tuberculosis', 'what is the sickness disease with the same kind of pathogen?', 'what are the sexually transmitted diseases carried by the same type of pathogen as tuberculosis?', 'what sex transmitted diseases are carried by the same type of broker as tuberculosis', 'what sex broadcast diseases constitute carry by the same type of broker as tuberculosis', 'what sex transfer diseases are transport by the same type of broker as tuberculosis', 'what sex transmitted diseases are pack by the same type of agent as tuberculosis', 'what sexuality transmitted disease are carried by the same type of broker as tuberculosis', 'what sex impart diseases are pack by the same type of agent as tuberculosis', 'what sexuality transmitted diseases are carried by the same type of broker as tuberculosis', 'what sex transfer diseases are transmit by the same type of agent as tuberculosis', 'what sexuality transfer disease constitute transport by the same type of broker as t.b.', 'what sex transmitted diseases are hold by the same type of agent as tuberculosis', 'what gender broadcast disease constitute carried by the same type of agent as tuberculosis', 'which sexually transmitted diseases carry the same type of pathogen as tuberculosis?', 'what sex transmit diseases are carried by the same type of agent as tuberculosis', 'what are the sexually transmitted diseases that have the same character like tuberculosis pathogens?', 'sexually transmitted diseases are vectors of tuberculosis of the same pathogen type.', 'sexual infections are infected with tuberculosis, which is the same pathogen.', 'what gender transmitted disease are carried by the same type of agent as tuberculosis', 'what sexuality transmitted diseases are carried by the same type of agent as tb', 'what sex send diseases be transport by the same type of agent as tuberculosis', 'what gender transfer disease are hold by the same type of agent as t.b.', 'what sexual activity transmitted disease are carried by the same character of agentive role as tuberculosis', 'what sex transmit disease are pack by the same type of broker as tuberculosis', 'what sex transmitted diseases are carried by the same type of agent as tb', 'what gender transmitted diseases are carried by the same type of broker as tuberculosis', 'what sex send diseases constitute carry by the same type of agent as tuberculosis', 'what sex transmitted diseases are impart by the same type of agent as tuberculosis', 'what sexual activity transfers the disease that has the same character as the pathogen of tuberculosis?', 'what gender transmitted disease are pack by the same type of agent as tuberculosis', 'what sex transmit disease constitute carried by the same type of agent as tb', 'what sex broadcast diseases are carried by the same type of agent as tuberculosis', 'which venereal diseases are transmitted by the same pathogen types as tuberculosis?', 'what are sexually transmitted diseases that play a role similar to that of tuberculosis?', 'sexually transmitted diseases are tuberculosis carriers of the same pathogen type.', 'what sex transmit diseases are carry by the same type of agent as tuberculosis', 'what sex transmitted diseases are carried by the same type of broker as tb', 'what sex transmitted diseases are transport by the same type of agent as tuberculosis', 'what sex transmitted disease are carried by the same type of agent as t.b.', 'which sexually transmitted diseases are transmitted by the same tuberculosis agent?', 'what sex send diseases constitute carried by the same type of agent as tuberculosis', 'which sexually transmitted diseases play a role similar to that of tuberculosis?', 'what sex transmit diseases be impart by the same type of agent as tuberculosis', 'what sexuality transmitted diseases constitute carried by the same type of agent as tuberculosis', 'sexually transmitted infections are transmitted along with tuberculosis, a pathogen of the same type.', 'what sex transmit diseases constitute hold by the same type of agent as tuberculosis', 'what sex transmit diseases constitute impart by the same type of agent as tuberculosis', 'what gender transmitted diseases are carried by the same type of agent as tuberculosis', 'what sex send disease be transport by the same type of broker as t.b.', 'what sexuality impart diseases are transmit by the same type of agent as tuberculosis', 'what gender impart disease be carried by the same type of broker as tuberculosis', 'what gender send disease constitute carried by the same type of broker as t.b.', 'what sexuality broadcast disease constitute transmit by the same type of agent as tuberculosis', 'what sex broadcast diseases constitute impart by the same type of agent as tuberculosis', 'what are the sexually transmitted diseases transmitted by the same type of pathogen as tuberculosis?', 'what sex transmit disease be carried by the same type of broker as tuberculosis', 'what sex transmitted diseases constitute carried by the same type of agent as tuberculosis', 'sexually transmitted diseases carry tuberculosis of the same type of drug', 'does the sexual activity of infectious disease have the same drug as the tuberculosis?', 'what sexuality transmitted diseases be transmit by the same type of broker as tuberculosis', 'what sexuality transmitted diseases are carried by the same type of agent as tuberculosis', 'what sex transfer diseases be transmit by the same type of agent as tuberculosis', 'what sexual activity transmit disease are carry by the same character of agentive role as tuberculosis', 'what sex transfer diseases are carry by the same type of agent as tuberculosis', 'what sex transmitted diseases are carried by the same type of agent as tuberculosis', 'like tuberculosis, what kind of sexual activity can transmit the disease?', 'what gender broadcast disease constitute transmit by the same type of agent as tuberculosis', 'what sexuality send diseases be hold by the same type of agent as tuberculosis'}}
    #universal sentence encoder filtering
    print("Start Universal Sentence Encoder filtering ",end="")
    t = time.time()
    use_filtered_paraphrases = use.get_embedding(data)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
    # write_to_folder(use_filtered_paraphrases,"Universal Sentence Encoder Filtering:","paraphrases.txt")

    print("Start BERT filtering ",end="")
    t = time.time()
    bert_filtered_paraphrases = bert.bert_selection(use_filtered_paraphrases)
    print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
    # write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")

    # sort the dictionary
    bert_filtered_paraphrases = sort_collection(bert_filtered_paraphrases)
    cut_off = 0
    if cut_off > 0:
        print("Start cut-off ",end="")
        final_result = apply_cut_off(bert_filtered_paraphrases,cut_off)
        print("\t- Elapsed time: ",str(datetime.timedelta(0,time.time()-t)))
        write_to_folder(final_result,"Final Paraphrases List:","paraphrases.txt")
    else:
        write_to_folder(bert_filtered_paraphrases,"Final Paraphrases List:","paraphrases.txt")
    
    paraphrases = bert_filtered_paraphrases
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
    
    # compute BLEU-Score of generated paraphrases
    print("Compute BLEU, GLEU and CHRF scores: ")
    bleu_score.main(paraphrases,cut_off)
    gleu_score.main(paraphrases,cut_off)
    chrf_score.main(paraphrases,cut_off)
    print("Overall elapsed time: ",str(datetime.timedelta(0,time.time()-t1)))

if __name__ == "__main__":
    main()
    