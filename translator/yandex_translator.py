import requests
import os
from yandex import Translater as yandex_translate
from translator import google_translator as google

""" This code translate sentence using Yandex Translator API """


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

def translate(utterance,source,target,api_key):
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
    rep = tr.translate()
    return rep.rstrip("\n")

def multiTranslate(utterance,api_key,pivot_level):
  """
  Translate sentence
  :param utterance: sentence to translate
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return list of utterance translations
  """
  response = set()
  text = utterance
  if pivot_level == 0 or pivot_level == 1:
    tmp = translate(text,'en','it',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'it','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ru',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ru','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ar',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ar','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','fr',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'fr','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ja',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ja','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','zh',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'zh','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','de',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'de','en',api_key)
    response.add(tmp)
    
  if pivot_level == 0 or pivot_level == 2:
    tmp = translate(text,'en','de',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'de','ru',api_key)
    tmp = translate(tmp,'ru','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','fr',api_key)
    tmp = translate(tmp,'fr','ru',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ru','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ar',api_key)
    tmp = translate(tmp,'ar','fr',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'fr','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','it',api_key)
    tmp = translate(tmp,'it','ru',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ru','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ru',api_key)
    tmp = translate(tmp,'ru','ar',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ar','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ru',api_key)
    tmp = translate(tmp,'ru','zh',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'zh','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ru',api_key)
    tmp = translate(tmp,'ru','ja',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ja','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ar',api_key)
    tmp = translate(tmp,'ar','zh',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'zh','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','ar',api_key)
    tmp = translate(tmp,'ar','ja',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'ja','en',api_key)
    response.add(tmp)

    tmp = translate(text,'en','de',api_key)
    tmp = translate(tmp,'de','fr',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'fr','en',api_key)
    response.add(tmp)
  return response


def translateFile(filePath,api_key,pivot_level):
  """
  Translate a file
  :param filePath: file path
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """

  paraphrases = dict()
  #import data from COVID19_data file
  f=open(filePath, "r")
  while True: 
      # Get next line from file 
      line = f.readline()
      if not line: 
          break
      tmp = multiTranslate(line,api_key,pivot_level)
      line = line.rstrip("\n")
      paraphrases[line]=tmp

  return paraphrases

def translateDict(data,api_key,pivot_level):
  """
  Translate a dictionary
  :param data: data in python dictionary, Key initial expression and value is a set of translations
  :param api_key: Yandex Translate API token https://translate.yandex.com/developers/keys
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """
  paraphrases = dict()
 
  for key,value in data.items():
    tmp = multiTranslate(value,api_key,pivot_level)
    paraphrases[key]=tmp
  return paraphrases
