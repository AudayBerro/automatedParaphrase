import requests
import os
from yandex import Translater as yandex_trans
import re,string
import contractions

"""" Thi code translate sentence using Mymemory API and Yandex Translator API """


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
    return (re.sub(' +', ' ',(text.replace('\n',' ')))).strip().lower()

def expand_contractions(text):
    """ expand shortened words, e.g. don't to do not """

    return contractions.fix(text)

def yandex_translate(utterance,source,target,api_key):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param source: source language
    :param target: target language
    :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
    :return Translated utterance 
    """

    tr = yandex_trans.Translater()
    tr.set_key(api_key) 
    tr.set_from_lang(source)
    tr.set_to_lang(target)
    tr.set_text(utterance) #text_to_translate
    response = tr.translate()
    return normalize_text(response)

def replace_quote(utterance):
    """
    Replace &quot; by \" and &#39 by \' returned in Yandex translation
    :return Utterance without Yandex quot tags
    """

    if "&quot;" in utterance:
      utterance = utterance.replace('&quot;','\"')
    if "&#39;" in utterance:
      utterance = utterance.replace('&#39;','\'')
    
    utterance = expand_contractions(utterance)
    return normalize_text(utterance)

def check_match(utterance):
    """
    Check if the quality of MyMemory API request translation
    :param utterance: translated sentence to check
    :return True - good translation - if translation above threshold, False - bad translaion -
    """

    threshold = 0.50
    if utterance['match'] < threshold: # ['match'] is the confidence score of the translation provided by MyMemory Translator model
        return False
    return True

def translate_file(file_path,valid_mail):
    """
    Translate a file
    :param file_path: file path
    :param valid_mail: valid email address to reach a translation rate of 10000 words/day in MyMemory API. https://mymemory.translated.net/doc/usagelimits.php
    :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
    """

    response = dict()
    f=open(file_path, "r")
    while True: 
        # Get next line from file 
        text = f.readline()
        if not text: 
            break

        sent = expand_contractions(text)
        sent = normalize_text(sent)

        # api-endpoint
        URL = "https://api.mymemory.translated.net/get?de="+valid_mail+"&q="+text+"&langpair=en|fr"

        # sending get request and saving the response as response object 
        r = requests.get(url = URL) 
        
        # extracting data in json format 
        data = r.json()

        tmp = data['matches']

        rep = set()#will contain all french translation
        rep.add(replace_quote(data['responseData']['translatedText']))
        for i in tmp:
            if check_match(i):
                sentence=replace_quote(i['translation'])
                rep.add(sentence)

        final = set()
        for sourceText in rep:
            URL = "https://api.mymemory.translated.net/get?de="+valid_mail+"&q="+sourceText+"&langpair=fr|en"
            # sending get request and saving the response as response object 
            r = requests.get(url = URL)
            data = r.json()
            
            tmp = data['matches']

            for i in tmp:
                sentence = replace_quote(i['translation'])
                if check_match(i):
                    final.add(sentence)
        text = text.replace('\n', '').replace('\r', '') #remove linebreak
        response[text]=final

    return response

def translate_list(data_set,valid_mail):
    """
    Translate a a list of sentences
    :param data_set: snetences to transalte in a form of python list
    :param valid_mail: valid email address to reach a translation rate of 10000 words/day in MyMemory API. https://mymemory.translated.net/doc/usagelimits.php
    :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
    """

    response = dict()
    for utterance in data_set:
        # api-endpoint
        URL = "https://api.mymemory.translated.net/get?de="+valid_mail+"&q="+utterance+"&langpair=en|fr"

        # sending get request and saving the response as response object 
        r = requests.get(url = URL) 
        
        # extracting data in json format 
        data = r.json()

        tmp = data['matches']

        rep = set()#will contain all french translation
        rep.add(replace_quote(data['responseData']['translatedText']))
        for i in tmp:
            if check_match(i):
                sentence=replace_quote(i['translation'])
                rep.add(sentence)

        final = set()
        for sourceText in rep:
            URL = "https://api.mymemory.translated.net/get?de="+valid_mail+"&q="+sourceText+"&langpair=fr|en"
            # sending get request and saving the response as response object 
            r = requests.get(url = URL)
            data = r.json()
            
            tmp = data['matches']

            for i in tmp:
                sentence = replace_quote(i['translation'])
                if check_match(i):
                    final.add(normalize_text(sentence))
        response[utterance] = final

    return response