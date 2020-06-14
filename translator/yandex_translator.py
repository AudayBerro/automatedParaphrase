import requests
import os
from yandex import Translater as yandex_translate
from translator import google_translator as google

""" This code translate sentence using Yandex Translator API """

def normalize_text(text):
    """
    Remove line break and lowercase all words
    :param text: sentence to normalize
    :return return a sentence without line break and lowercased 
    """
    return text.replace('\n', ' ').replace('\r', '').lower()

def replace_quote(utterance):
    """
    Replace &quot; by \" and &#39 by \' returned in Yandex translation
    :return Utterance without Yandex quot tags
    """

    if "&quot;" in utterance:
      utterance = utterance.replace('&quot;','\"')
    if "&#39;" in utterance:
      utterance = utterance.replace('&#39;','\'')
    
    return normalize_text(utterance)

def translate(utterance,source,target,tr):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param source: source language
    :param target: target language
    :param tr: yandex Translator object
    :return Translated utterance 
    """

    tr.set_from_lang(source)
    tr.set_to_lang(target)
    tr.set_text(utterance) #text_to_translate
    rep = tr.translate()
    return normalize_text(rep)

def multi_translate(utterance,api_key,pivot_level):
  """
  Translate sentence
  :param utterance: sentence to translate
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return list of utterance translations
  """

  tr = yandex_translate.Translater() # load yandex translator
  tr.set_key(api_key) # set the api token

  response = set()
  text = utterance
  if pivot_level == 0 or pivot_level == 1:
    tmp = translate(text,'en','it',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'it','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ru',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ru','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ar',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ar','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','fr',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'fr','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ja',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ja','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','zh',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'zh','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','de',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'de','en',tr)
    response.add(tmp)
    
  if pivot_level == 0 or pivot_level == 2:
    tmp = translate(text,'en','de',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'de','ru',tr)
    tmp = translate(tmp,'ru','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','fr',tr)
    tmp = translate(tmp,'fr','ru',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ru','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ar',tr)
    tmp = translate(tmp,'ar','fr',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'fr','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','it',tr)
    tmp = translate(tmp,'it','ru',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ru','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ru',tr)
    tmp = translate(tmp,'ru','ar',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ar','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ru',tr)
    tmp = translate(tmp,'ru','zh',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'zh','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ru',tr)
    tmp = translate(tmp,'ru','ja',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ja','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ar',tr)
    tmp = translate(tmp,'ar','zh',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'zh','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','ar',tr)
    tmp = translate(tmp,'ar','ja',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ja','en',tr)
    response.add(tmp)

    tmp = translate(text,'en','de',tr)
    tmp = translate(tmp,'de','fr',tr)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'fr','en',tr)
    response.add(tmp)
  return response


def translate_file(file_path,api_key,pivot_level):
  """
  Translate a file
  :param file_path: file path
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """

  paraphrases = dict()
  #import data from file_path
  f=open(file_path, "r")
  while True: 
      # Get next line from file 
      line = f.readline()
      if not line: 
          break
      line = normalize_text(line)
      tmp = multi_translate(line,api_key,pivot_level)
      paraphrases[line]=tmp

  return paraphrases

def translate_dict(data,api_key,pivot_level):
  """
  Translate a dictionary
  :param data: data in python dictionary, Key initial expression and value is a set of translations
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """
  paraphrases = dict()
 
  for key,value in data.items():
    tmp = multi_translate(value,api_key,pivot_level)
    paraphrases[key]=tmp
  return paraphrases

def translate_list(data,api_key,pivot_level):
  """
  Translate a List of sentences
  :param data: data in python List, list of sentences
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """
  paraphrases = dict()
 
  for sentence in data:
    tmp = multi_translate(sentence,api_key,pivot_level)
    paraphrases[sentence]=tmp
  return paraphrases