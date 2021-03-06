import requests
from googletrans import Translator
import re,string
import contractions

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

def expand_contractions(text):
    """ expand shortened words, e.g. don't to do not """

    return contractions.fix(text)

def translate_wrapper(sentence,target):
    """
    Translate sentence to target language using googletrans library which is a Google Translate API wrapper,
    call this function when you don't have Google Translate API credentials
    :param sentence: sentence to translate
    :param target: target language value can be: ar,fr,en,de,ru,zh,ja,it,... visit for more language https://cloud.com/translate/docs/languages
    :return translated sentence
    """
    translator = Translator()
    
    try:
        # translate the 'text' column
        response = translator.translate(sentence, dest=target)

    except Exception as e: # mean Google restrict IP address
        response = "Probably Google has banned your client IP addres"+str(e)
        return response
    tmp = expand_contractions(response.text)
    return normalize_text(tmp)
    # return response.text


def translate(utterance,target,api_key):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param target: target language
    :param api_key: Authentication Key for DeepL API https://www.deepl.com/pro-account.html
    :return Translated utterance 
    """

    data = {
    'auth_key': api_key,
    'text': utterance,
    'target_lang': target
    }

    response = requests.post('https://api.deepl.com/v2/translate', data=data)
    data = response.json()
    return data['translations'][0]['text']

def multi_translate(utterance,api_key,pivot_level):
  """
  Translate sentence
  :param utterance: sentence to translate
  :param api_key: Authentication Key for DeepL API https://www.deepl.com/pro-account.html
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return list of utterance translations
  """
  
  response = set()
  text = utterance
  if pivot_level == 0 or pivot_level == 1:
    tmp = translate(text,'IT',api_key)
    response.add(translate_wrapper(tmp,'en')) # translate back to english with google translator wrapper
    # tmp = translate(tmp,'EN',api_key) # translate back to english with deepl
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))
    

    tmp = translate(text,'RU',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'FR',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'JA',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'DE',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))
    
  if pivot_level == 0 or pivot_level == 2:
    tmp = translate(text,'IT',api_key)
    tmp = translate(tmp,'RU',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'IT',api_key)
    tmp = translate(tmp,'DE',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'RU',api_key)
    tmp = translate(tmp,'FR',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'RU',api_key)
    tmp = translate(tmp,'JA',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'JA',api_key)
    tmp = translate(tmp,'FR',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))

    tmp = translate(text,'JA',api_key)
    tmp = translate(tmp,'DE',api_key)
    response.add(translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    tmp = expand_contractions(tmp)
    response.add(normalize_text(tmp))
  return response

def translate_file(file_path,api_key,pivot_level):
  """
  Translate a file
  :param file_path: file path
  :param api_key: Authentication Key for DeepL API https://www.deepl.com/pro-account.html
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
      line = line.replace('\n', '').replace('\r', '') #remove linebreak
      tmp = multi_translate(line,api_key,pivot_level)
      paraphrases[line]=tmp

  return paraphrases

def translate_dict(data,api_key,pivot_level):
  """
  Translate a dictionary
  :param data: data in python dictionary, Key initial expression and value is a set of translations
  :param api_key: Authentication Key for DeepL API https://www.deepl.com/pro-account.html
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
  :param api_key: Authentication Key for DeepL API https://www.deepl.com/pro-account.html
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """
  paraphrases = dict()
 
  for sentence in data:
    tmp = multi_translate(sentence,api_key,pivot_level)
    paraphrases[sentence]=tmp
  return paraphrases

if __name__ == "__main__":
    print(multi_translate('How does COVID-19 spread?','f55c628f-a052-4431-90a7-86d0b8ca861b',1))
    
