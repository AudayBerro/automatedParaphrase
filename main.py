from translator import my_memory_translator as memory
# from translator import yandex_translator as yandex
from translator import deepl_translator as deepl
from translator import marian_translator as marian
from pos import pos_extraction as pos
from filtering import bert_filter as bert
from filtering import use_filter as use
from synonym import nltk_wordnet as nlt
import os
import configparser
#import spacy

import argparse
import re,string

def normalize_text(text):
    """
    Remove line break and lowercase all words
    :param text: sentence to normalize
    :return return a sentence without line break and lowercased 
    """
    # return text.replace('\n', ' ').replace('\r', '').lower()
    #remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    #trim and lowercase
    return (re.sub(' +', ' ',(text.replace('\n','')))).strip().lower()

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


def weak_supervision_generation(file_path):
    """
    Apply Weak Supervision to generate data
    :param file_path: file path to folder containing initial utterances
    :return generated data 
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


def online_transaltion(file_path,api_key,valid_mail,pivot_level):
    """
    Generate Paraphrases Using online Translator Engine e.g. Google, Yandex
    :param file_path: file path to folder containing initial utterances
    :param api_key: Online Translator API key
    :param valid_mail: valid email address to reach a translation rate of 10000 words/day in MyMemory API.
    :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
    :return a Python dictionary, Key is the initial expression and value is a list of paraphrases
    """
    #wordnet
    print("Start weak supervision data generation")
    data1,data2,data3 = weak_supervision_generation(file_path)

    print("Start translation")
    # generate paraphrases with MyMemory API
    memory_result1 = memory.translate_list(data1,valid_mail) #generate paraphrases through pivot-translation of data1
    memory_result2 = memory.translate_list(data2,valid_mail) #generate paraphrases through pivot-translation of data2
    memory_result3 = memory.translate_list(data3,valid_mail) #generate paraphrases through pivot-translation of data3
    
    result = memory.translate_file(file_path,valid_mail) #generate paraphrases through pivot-translation of initial utterances folder

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

    write_to_folder(result,"Generated Paraphrases:","paraphrases.txt")
    #universal sentence encoder filtering
    print("Start Universal Sentence Encoder filtering")
    use_filtered_paraphrases = use.get_embedding(result)
    write_to_folder(use_filtered_paraphrases,"Universal Sentence Encoder Filtering:","paraphrases.txt")

    # apply BERT filtering after USE filtering
    print("Start BERT filtering")
    bert_filtered_paraphrases = bert.bert_filtering(use_filtered_paraphrases)
    write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")
    print("Start BERT deduplication")
    bert_deduplicate_paraphrases = bert.bert_deduplication(bert_filtered_paraphrases)
    write_to_folder(bert_deduplicate_paraphrases,"BERT deduplication:","paraphrases.txt")

def pretrained_transaltion(file_path,pivot_level):
    """
    Generate Paraphrases using Pretrained Translation Model e.g. Huggingface MarianMT
    :param file_path: file path to folder containing initial utterances
    :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
    :return a Python dictionary, Key is the initial expression and value is a list of paraphrases
    """
    #load all the model
    # print("load model")
    model_list = marian.load_model()
    
    #wordnet
    print("Start weak supervision data generation")
    data1,data2,data3 = weak_supervision_generation(file_path)

    print("Start translation")
    # generate paraphrases with MyMemory API
    result1 = marian.translate_list(data1,model_list,pivot_level)
    result2 = marian.translate_list(data2,model_list,pivot_level)
    result3 = marian.translate_list(data3,model_list,pivot_level)
    
    result = marian.translate_file(file_path,model_list,pivot_level) #  (file_path,model_list,pivot_level)

    # merge result1, result2, result3 with result
    result= merge_data(result,result1)
    result= merge_data(result,result2)
    result= merge_data(result,result3)

    write_to_folder(result,"Generated Paraphrases:","paraphrases.txt")
    #universal sentence encoder filtering
    print("Start Universal Sentence Encoder filtering")
    use_filtered_paraphrases = use.get_embedding(result)
    write_to_folder(use_filtered_paraphrases,"Universal Sentence Encoder Filtering:","paraphrases.txt")
    # bert_filtered_paraphrases = bert.bert_filtering(use_filtered_paraphrases)
    # write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")
    # print("Start BERT deduplication")
    # bert_deduplicate_paraphrases = bert.bert_deduplication(bert_filtered_paraphrases)
    # write_to_folder(bert_deduplicate_paraphrases,"BERT deduplication:","paraphrases.txt")
    print("Start BERT filtering")
    bert_filtered_paraphrases = bert.bert_selection(use_filtered_paraphrases)
    write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")

def main():
    # required arg
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True) # -f data set file name argument
    parser.add_argument('-g') # if -g is defined use google_translator.translate method not translate_wrapper
    parser.add_argument('-l') # -l integer that indicate the pivot language level, single-pivot or multi-pivot range between 0 and 2
    parser.add_argument('-p') # use pretrained translator(p==true - MarianMT) or online translator engine(p==false - Yandex,Google Translator)
    parser.add_argument('-s') # cut-off criteria to stop paraphrasing, default s=5
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
        
        if args.s:
            cut_off = int(args.s)
            if cut_off<=0:
                raise Exception("Cut-off parameter value should be greater or equal to 1")
        else:
            cutt_off = 5 # default value

    except Exception as e:
        print(str(e))
        exit()

    file_path = os.path.join(os.path.dirname(__file__), ".", "dataset/"+args.f) # data to paraphrase
    if args.p=="true":
        pretrained_transaltion(file_path,pivot_level)
    else:
        online_transaltion(file_path,deepl_api_key,valid_mail,pivot_level)

if __name__ == "__main__":
    main()