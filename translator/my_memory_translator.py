import requests
import os
from yandex import Translater as yandex_translate

"""" Thi code translate sentence using Mymemory API and Yandex Translator API """

def yandexTranslate(utterance,source,target,api_key):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param source: source language
    :param target: target language
    :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
    :return Translated utterance 
    """

    tr = yandex_translate.Translater()
    tr.set_key(api_key) 
    tr.set_from_lang(source)
    tr.set_to_lang(target)
    tr.set_text(utterance) #text_to_translate
    response = tr.translate()
    return response.rstrip("\n")

def replaceQuote(utterance):
    """
    Replace &quot; by \" and &#39 by \' returned in Yandex translation
    :return Utterance without Yandex quot tags
    """

    if "&quot;" in utterance:
      utterance = utterance.replace('&quot;','\"')
    if "&#39;" in utterance:
      utterance = utterance.replace('&#39;','\'')
    
    return utterance

def checkMatch(utterance):
    """
    Check if the quality of MyMemory API request translation
    :param utterance: translated sentence to check
    :return True - good translation - if translation above threshold, False - bad translaion -
    """

    threshold = 0.50
    if utterance['match'] < threshold: # ['match'] is the confidence score of the translation provided by MyMemory Translator model
        return False
    return True

def translateFile(filePath,valid_mail):
    """
    Translate a file
    :param filePath: file path
    :param valid_mail: valid email address to reach a translation rate of 10000 words/day in MyMemory API. https://mymemory.translated.net/doc/usagelimits.php
    :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
    """

    response = dict()
    f=open(filePath, "r")
    while True: 
        # Get next line from file 
        text = f.readline()
        if not text: 
            break

        # api-endpoint 
        URL = "https://api.mymemory.translated.net/get?de="+valid_mail+"&q="+text+"&langpair=en|fr"

        # sending get request and saving the response as response object 
        r = requests.get(url = URL) 
        
        # extracting data in json format 
        data = r.json()

        tmp = data['matches']

        rep = set()#will contain all french translation
        rep.add(replaceQuote(data['responseData']['translatedText']))
        for i in tmp:
            if checkMatch(i):
                sentence=replaceQuote(i['translation'])
                rep.add(sentence)

        final = set()
        for sourceText in rep:
            #    yanTrans = yandexTranslate(sentence,'fr','en')
            #    final.add(yanTrans)
            URL = "https://api.mymemory.translated.net/get?de="+valid_mail+"&q="+sourceText+"&langpair=fr|en"
            # sending get request and saving the response as response object 
            r = requests.get(url = URL)
            data = r.json()
            
            tmp = data['matches']

            for i in tmp:
                sentence = replaceQuote(i['translation'])
                if checkMatch(i):
                    final.add(sentence.rstrip("\n"))
        response[text.rstrip("\n")]=final

    return response
