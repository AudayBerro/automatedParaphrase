from translator import my_memory_translator as memory
from translator import yandex_translator as yandex
from pos import pos_extraction as pos
from filtering import bert_filter as bert
from filtering import use_filter as use
import os
import configparser
#import spacy

import argparse


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

def main():
    # required arg
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True) # -f data set file name argument
    parser.add_argument('-g') # if -g is defined use google_translator.translate method not translate_wrapper
    parser.add_argument('-l') # -l integer that indicate the pivot language level, single-pivot or multi-pivot range between 1 and 3
    args = parser.parse_args()
    
    # load configs from config.ini file
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(os.path.join(os.path.dirname(__file__), ".","config.ini"))
    my_memory_config = config["MYMEMORY"]
    yandex_config = config["YANDEX"]
    google_config = config["GOOGLE"]

    try:
        if "email" not in my_memory_config or my_memory_config["email"] == "":
            raise Exception("Define a Valid email address for MyMemory API in config.ini")
        else:
            valid_mail = my_memory_config['email']
        if "api_key" not in yandex_config or yandex_config["api_key"] == "":
            raise Exception("Yandex Translate API token is not defined in config.ini")
        else:
            yandex_api_key = yandex_config["api_key"]
        if args.g:
            if "api_key" not in google_config or google_config["api_key"] == "":
                raise Exception("Google Translate API token is not defined in config.ini")
            else:
                google_api_key = google_config['api_key']
        if args.l:
            pivot_level = int(args.l)
        else:
            pivot_level = 0

    except Exception as e:
        print(str(e))
        exit()

    file_path = os.path.join(os.path.dirname(__file__), ".", "dataset/"+args.f) # data to paraphrase
    print("Start translation")
    result = memory.translateFile(file_path,valid_mail)

    yandexResult = yandex.translateFile(file_path,yandex_api_key,pivot_level)

    extracted_pos = pos.postExtraction(file_path)
    yandex_paraphrases = yandex.translateDict(extracted_pos,yandex_api_key,pivot_level)

    for key,values in result.items():
        values.update(yandexResult[key])
        values.update(yandex_paraphrases[key])
        result[key] = values


    write_to_folder(result,"Generated Paraphrases:","paraphrases.txt")
    #universal sentence encoder filtering
    print("Start Universal Sentence ENcoder filtering")
    use_filtered_paraphrases = use.get_embedding(result)
    write_to_folder(use_filtered_paraphrases,"Universal Sentence Encoder Filtering:","paraphrases.txt")

    # apply BERT filtering after USE filtering
    print("Start BERT filtering")
    bert_filtered_paraphrases = bert.bert_filtering(use_filtered_paraphrases)
    write_to_folder(bert_filtered_paraphrases,"BERT filtering:","paraphrases.txt")
    print("Start BERT deduplication")
    bert_deduplicate_paraphrases = bert.bert_deduplication(bert_filtered_paraphrases)
    write_to_folder(bert_deduplicate_paraphrases,"BERT deduplication:","paraphrases.txt")

if __name__ == "__main__":
    main()
