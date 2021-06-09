import contractions
import re,string,time
import concurrent.futures

from transformers import MarianMTModel,MarianTokenizer
from translator import easy_nmt

""" This code translate sentence using Huggingface Marian Machine Translation Pretrained Model """

def pr_gray(msg):
    """ Pring msg in gray color font"""
    print("\033[7m{}\033[00m" .format(msg))

def pr_green(msg):
    """ Pring msg in green color font"""
    print("\033[92m{}\033[00m" .format(msg))

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


def translate(utterance,model,tok,trg="NONE"):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param model: transformers Marian Machine Transaltion Model(MarianMTModel)
    :param tok: transformers Marian Tokenizer module(MarianTokenizer)
    :param trg: target language - set value when using en-ROMANCE model - trg=>>fr<<|>>it<<|>>es<<|>>pt<<
    :return Translated utterance 
    """
    if trg != 'NONE':
        utterance = '>>'+trg+'<<  '+utterance
    # translated = model.generate(**tok.prepare_translation_batch([utterance]))#old version transformers==3.0.0
    translated = model.generate(**tok(utterance, return_tensors="pt", padding=True))
    result = [tok.decode(t, skip_special_tokens=True) for t in translated]

    result = result[0]

    # check token indices sequence length is longer than the specified maximum sequence length max_length=512
    if len(result) > 512:
      result = result[:512]
    return result


def multi_translate(utterance,model,pivot_level=1):
  """
  Translate sentence
  :param utterance: sentence to translate
  :param model_list: dictionary containing marianMT model, key: model name - value: list containing respectively  Model and tokenizer.  e.g. {'en2ROMANCE':[model,tekenizer]}
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return list of utterance translations
  """
  response = set()
  if pivot_level == 0 or pivot_level == 1:#one pivot language

    # Translate to Romance language: ['French fr','Spanish es','Italian it','Portuguese pt','Romanian ro']
    # Translate to Italian
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="it")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to French
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="fr")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Spanish
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Portuguese
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="pt")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Romanian
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="ro")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to German
    tmp = translate(utterance,model['en2de'][0],model['en2de'][1])
    tmp = translate(tmp,model['de2en'][0],model['de2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Russian
    tmp = translate(utterance,model['en2ru'][0],model['en2ru'][1])
    tmp = translate(tmp,model['ru2en'][0],model['ru2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Arabic
    tmp = translate(utterance,model['en2ar'][0],model['en2ar'][1])
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'en', 'ar') # translate back to English with EasyNMt 
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Chinese
    tmp = translate(utterance,model['en2zh'][0],model['en2zh'][1])
    tmp = translate(tmp,model['zh2en'][0],model['zh2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Japanese
    tmp = translate(utterance,model['en2jap'][0],model['en2jap'][1])
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'en', 'ja') # translate back to English with EasyNMt 
    response.add(tmp)#add the generated paraphrase candidate
    
  if pivot_level == 0 or pivot_level == 2:# two pivot language
    # Translate Spanish => Russian = > English
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'ru', 'es') # translate to Russian with EasyNMt
    tmp = translate(tmp,model['ru2en'][0],model['ru2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Japanese => Spanish = > English
    tmp = translate(utterance,model['en2jap'][0],model['en2jap'][1])#translate to Japanese
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'es', 'ja') # translate to Spanish with EasyNMt
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Japanese => Italian = > English
    tmp = translate(utterance,model['en2jap'][0],model['en2jap'][1])#translate to Japanese
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'it', 'ja') # translate to Italian with EasyNMt
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Arabic => German = > English
    tmp = translate(utterance,model['en2ar'][0],model['en2ar'][1])#translate to Arabic
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'de', 'ar') # translate to German with EasyNMt 
    tmp = translate(tmp,model['de2en'][0],model['de2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Chinese => German = > English
    tmp = translate(utterance,model['en2zh'][0],model['en2zh'][1])#translate to Chinese
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'de', 'zh') # translate to German with EasyNMt 
    tmp = translate(tmp,model['de2en'][0],model['de2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate German => Arabic = > English
    tmp = translate(utterance,model['en2de'][0],model['en2de'][1])#translate to German
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'ar', 'de') # translate to Arabic with EasyNMt 
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'en', 'ar') # translate to English with EasyNMt
    response.add(tmp)#add the generated paraphrase candidate

    # Translate German => Chinese = > English
    tmp = translate(utterance,model['en2de'][0],model['en2de'][1])#translate to German
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'zh', 'de') # translate to Chinese with EasyNMt 
    tmp = translate(tmp,model['zh2en'][0],model['zh2en'][1])# translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate German => Japanese = > English
    tmp = translate(utterance,model['en2de'][0],model['en2de'][1])#translate to German
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'ja', 'de') # translate to Chinese with EasyNMt 
    tmp = easy_nmt.get_translation( tmp, model['easy_nmt'], 'en', 'ja') # translate to English with EasyNMt 
    response.add(tmp)#add the generated paraphrase candidate

  return list(response)#convert to list

def translate_file(file_path,model,pivot_level):
  """
  Translate a file
  :param file_path: file path
  :param model_list: dictionary containing marianMT model, key: model name - value: list containing respectively  Model and tokenizer.  e.g. {'en2ROMANCE':[model,tekenizer]}
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
      
      tmp = expand_contractions(line)
      tmp = normalize_text(tmp)
      line = line.replace('\n', '').replace('\r', '') #remove linebreak
      paraphrases[line]=multi_translate(tmp,model,0)

  return paraphrases

def translate_list(data,model,pivot_level):
  """
  Translate a List of sentences
  :param data: data in python List, list of sentences
  :param model_list: dictionary containing marianMT model, key: model name - value: list containing respectively  Model and tokenizer.  e.g. {'en2ROMANCE':[model,tekenizer]}
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return Python dictionary containing translsation, Key are initial sentence and vaule are a set of translations
  """
  paraphrases = dict()
  for sentence in data:
    tmp = multi_translate(sentence,model,pivot_level)
    paraphrases[sentence]=tmp
  return paraphrases

def get_model(param):
  """
  Load Hugginface marian Machine Translator model and tokenizer
  :param param: Huggingface MarianMt Helsinki-NLP/{model_name} to load (https://huggingface.co/Helsinki-NLP); param[0]=label - param[1]=model_name
  :return a tuple result = (Huggingface MarianMt Model, Marian MT Tokenizer, Marian MT label)
  """
  mt_model = MarianMTModel.from_pretrained(param[1]) #param[0]=label ; param[1]=model_name to load
  mt_tokenizer = MarianTokenizer.from_pretrained(param[1]) #load tokenizer
  return mt_model,mt_tokenizer,param[0]

def concurrent_model_loader():
  """
  Return a List of Huggingface Marian MT model, same as load_model but load concurrently
  :return Python dictionary - key: model name - value: list containing respectively MarianModel and MarianTokenizer e.g. {'en2ru':[model,tokenizer]}
  """
  response = dict()

  # set containing list of model to load with their respective label
  models_to_load = {
    ('en2romance','Helsinki-NLP/opus-mt-en-ROMANCE'),
    ('romance2en','Helsinki-NLP/opus-mt-ROMANCE-en'),
    ('de2en','Helsinki-NLP/opus-mt-de-en'),
    ('ru2en','Helsinki-NLP/opus-mt-ru-en'),
    ('en2ar','Helsinki-NLP/opus-mt-en-ar'),
    ('en2zh','Helsinki-NLP/opus-mt-en-zh'),
    ('en2jap','Helsinki-NLP/opus-mt-en-jap'),
    ('en2ru','Helsinki-NLP/opus-mt-en-ru'),
    ('en2de','Helsinki-NLP/opus-mt-en-de'),
    ('zh2en','Helsinki-NLP/opus-mt-zh-en')
  }

  # load HuggingFace Marian MT model and tokenizer concurrently through thread 
  with concurrent.futures.ThreadPoolExecutor() as executor:

    # results = [executor.submit(get_model2,model_name) for model_name in models_to_load.values()]
    results = executor.map( get_model, models_to_load)

    # unpack and add MarianMT model, MarianMT tokenizer and label
    for model,tokenizer,label in results:
        response[label] = [model,tokenizer]
  
  #load EasyNMT nodel
  mname = 'm2m_100_418M'
  easy_model = easy_nmt.load_model(mname)
  response['easy_nmt'] = easy_model
  return response

def load_model():
    """
    Return a List of Huggingface Marian MT model
    :return Python dictionary - key: model name - value: list containing respectively MarianModel and MarianTokenizer e.g. {'en2ru':[model,tokenizer]}
    """

    response = dict()

    # load HuggingFace Marian MT model and tokenizer
    # English to Romance: ['French fr','Spanish es','Italian it','Portuguese pt','Romanian ro']
    pr_green("\nLoad Huggingface Marian MT model:")
    pr_gray("\tload English to Romance model")
    mname = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    en2romance_model = MarianMTModel.from_pretrained(mname) #load model
    en2romance_tok = MarianTokenizer.from_pretrained(mname) #load tokenizer
    response['en2romance']=[en2romance_model,en2romance_tok]

    # Romance to English: ['French','Spanish','Italian','Portuguese']
    pr_gray("\tload Romance to English model")
    mname = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    romance_en_model = MarianMTModel.from_pretrained(mname) #load model
    romance_en_tok = MarianTokenizer.from_pretrained(mname) #load tokenizer
    response['romance2en']=[romance_en_model,romance_en_tok]

    # load German to English model
    mname = 'Helsinki-NLP/opus-mt-de-en'
    pr_gray("\tload German to English model")
    de2en_model = MarianMTModel.from_pretrained(mname)
    de2en_tok = MarianTokenizer.from_pretrained(mname)
    response['de2en']=[de2en_model,de2en_tok]

    # load russian to english model
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    pr_gray("\tload Russian to English model")
    ru2en_model = MarianMTModel.from_pretrained(mname)
    ru2en_tok = MarianTokenizer.from_pretrained(mname)
    response['ru2en']=[ru2en_model,ru2en_tok]

    # load English to Arabic model
    mname = 'Helsinki-NLP/opus-mt-en-ar'
    pr_gray("\tload English to Arabic model")
    en2ar_model = MarianMTModel.from_pretrained(mname)
    en2ar_tok = MarianTokenizer.from_pretrained(mname)
    response['en2ar']=[en2ar_model,en2ar_tok]

    # load English to Chinese model
    mname = 'Helsinki-NLP/opus-mt-en-zh'
    pr_gray("\tload English to Chinese model")
    en2zh_model = MarianMTModel.from_pretrained(mname)
    en2zh_tok = MarianTokenizer.from_pretrained(mname)
    response['en2zh']=[en2zh_model,en2zh_tok]

    # load English to Japanese model
    mname = 'Helsinki-NLP/opus-mt-en-jap'
    pr_gray("\tload English to Japanese model")
    en2jap_model = MarianMTModel.from_pretrained(mname)
    en2jap_tok = MarianTokenizer.from_pretrained(mname)
    response['en2jap']=[en2jap_model,en2jap_tok]

    # if pivot_level == 0 or pivot_level==1:#one pivot language
    # load english to russian model
    mname = 'Helsinki-NLP/opus-mt-en-ru'
    pr_gray("\tload English to Russian model")
    en2ru_model = MarianMTModel.from_pretrained(mname)
    en2ru_tok = MarianTokenizer.from_pretrained(mname)
    response['en2ru']=[en2ru_model,en2ru_tok]

    # load English to German model
    mname = 'Helsinki-NLP/opus-mt-en-de'
    pr_gray("\tload English to German model")
    en2de_model = MarianMTModel.from_pretrained(mname)
    en2de_tok = MarianTokenizer.from_pretrained(mname)
    response['en2de']=[en2de_model,en2de_tok]
    
    # load Chinese to English model
    mname = 'Helsinki-NLP/opus-mt-zh-en'
    pr_gray("\tload Chinese to English model")
    zh2en_model = MarianMTModel.from_pretrained(mname)
    zh2en_tok = MarianTokenizer.from_pretrained(mname)
    response['zh2en']=[zh2en_model,zh2en_tok]
    
    # load EasyNMT model
    mname = 'm2m_100_418M'
    easy_model = easy_nmt.load_model(mname)
    response['easy_nmt'] = easy_model

    return response

def main(model_list):
  #load all the model
  print("load model")
  file_path = os.path.join(os.path.dirname(__file__), "..", "dataset/dataset.txt") # data to paraphrase
  print("start translation")
  paraphrases = translate_file(file_path,model_list,1)
  print("Result:")
  for key,value in paraphrases.items():
    print(key)
    for i in value:
      print("\t",i)
      
if __name__ == "__main__":
  import os
  #load all the model
  t1 = time.perf_counter()
  print("load model")
  model_list = concurrent_model_loader()
  t2 = time.perf_counter()
  print(f'Finished in {round(t2-t1,2)} second(s)')

  sentence = "book a flight from Lyon to Sydney"
  paraphrases = multi_translate(sentence,model_list,0)
  print("Result:")
  for key in paraphrases:
    print(key)
  import sys
  sys.exit()
  print(len(model_list))
  main(model_list)
  dataset = ['How does COVID-19 spread?','Book a flight from lyon to sydney?','Reserve a Restaurant at Paris']
  print("start translation")
  paraphrases = translate_list(dataset,model_list,1)
  print("Result:")
  for key,value in paraphrases.items():
    print(key)
    for i in value:
      print("\t",i)
  
  # # model_name1 = 'Helsinki-NLP/opus-mt-en-mul'
  # # model_name2 = 'Helsinki-NLP/opus-mt-mul-en'

  # # mt_model1 = MarianMTModel.from_pretrained(model_name1) #param[0]=label ; param[1]=model_name to load
  # # mt_tokenizer1 = MarianTokenizer.from_pretrained(model_name1) #load tokenizer

  # # mt_model2 = MarianMTModel.from_pretrained(model_name2) #param[0]=label ; param[1]=model_name to load
  # # mt_tokenizer2 = MarianTokenizer.from_pretrained(model_name2) #load tokenizer

  # # utterance = "book a flight from Lyon to Sydney?"
  # # tmp = translate(utterance,mt_model1,mt_tokenizer1,trg="fra")
  # # print(f"fr: {tmp}")
  # # tmp = translate(tmp,mt_model2,mt_tokenizer2)#translate back to English
  # # print(f"en: {tmp}")

  # #EasyNMT module test