from transformers import MarianMTModel,MarianTokenizer
#from translator import open_nmt
import os
import contractions
import re,string

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
    return result[0]


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
    tmp = translate(tmp,model['ar2en'][0],model['ar2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Chinese
    tmp = translate(utterance,model['en2zh'][0],model['en2zh'][1])
    tmp = translate(tmp,model['zh2en'][0],model['zh2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate to Japanese
    tmp = translate(utterance,model['en2jap'][0],model['en2jap'][1])
    tmp = translate(tmp,model['jap2en'][0],model['jap2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate
    
  if pivot_level == 0 or pivot_level == 2:# two pivot language

    # Translate Spanish => Russian = > English
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    tmp = translate(tmp,model['es2ru'][0],model['es2ru'][1])#translate to russian
    tmp = translate(tmp,model['ru2en'][0],model['ru2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Japanese => Spanish = > English
    tmp = translate(utterance,model['en2jap'][0],model['en2jap'][1])#translate to Japanese
    tmp = translate(tmp,model['jap2es'][0],model['jap2es'][1])#translate to Spanish
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Arabic => German = > English
    tmp = translate(utterance,model['en2ar'][0],model['en2ar'][1])#translate to Arabic
    tmp = translate(tmp,model['ar2de'][0],model['ar2de'][1])#translate to German
    tmp = translate(tmp,model['de2en'][0],model['de2en'][1])#translate back to English
    response.add(tmp)#add the generated paraphrase candidate

    # Translate Chinese => German = > English
    tmp = translate(utterance,model['en2zh'][0],model['en2zh'][1])#translate to Chinese
    tmp = translate(tmp,model['zh2de'][0],model['zh2de'][1])#translate to German
    tmp = translate(tmp,model['de2en'][0],model['de2en'][1])#translate back to English
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

def load_model(pivot_level=1):
    """
    Return a List of Huggingface Marian MT model
    :param pivot_level: integer, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
    :return Python dictionary - key: model name - value: list containing respectively MarianModel and MarianTokenizer e.g. {'en2ru':[model,tokenizer]}
    """
    response = dict()

    #Required model used in One-Pivot and Two-Pivot
    #English to Romance: ['French fr','Spanish es','Italian it','Portuguese pt','Romanian ro']
    pr_green("\nLoad Huggingface Marian MT model:")
    pr_gray("\tload English to Romance model")
    mname1 = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    en2romance_model = MarianMTModel.from_pretrained(mname1) #load model
    en2romance_tok = MarianTokenizer.from_pretrained(mname1) #load tokenizer
    response['en2romance']=[en2romance_model,en2romance_tok]

    #Romance to English: ['French','Spanish','Italian','Portuguese']
    pr_gray("\tload Romance to English model")
    mname2 = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    romance_en_model = MarianMTModel.from_pretrained(mname2) #load model
    romance_en_tok = MarianTokenizer.from_pretrained(mname2) #load tokenizer
    response['romance2en']=[romance_en_model,romance_en_tok]

    #load German to English model
    mname = 'Helsinki-NLP/opus-mt-de-en'
    pr_gray("\tload German to English model")
    de2en_model = MarianMTModel.from_pretrained(mname)
    de2en_tok = MarianTokenizer.from_pretrained(mname)
    response['de2en']=[de2en_model,de2en_tok]

    #load russian to english model
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    pr_gray("\tload Russian to English model")
    ru2en_model = MarianMTModel.from_pretrained(mname)
    ru2en_tok = MarianTokenizer.from_pretrained(mname)
    response['ru2en']=[ru2en_model,ru2en_tok]

    #load English to Arabic model
    mname = 'Helsinki-NLP/opus-mt-en-ar'
    pr_gray("\tload English to Arabic model")
    en2ar_model = MarianMTModel.from_pretrained(mname)
    en2ar_tok = MarianTokenizer.from_pretrained(mname)
    response['en2ar']=[en2ar_model,en2ar_tok]

    #load English to Chinese model
    mname = 'Helsinki-NLP/opus-mt-en-zh'
    pr_gray("\tload English to Chinese model")
    en2zh_model = MarianMTModel.from_pretrained(mname)
    en2zh_tok = MarianTokenizer.from_pretrained(mname)
    response['en2zh']=[en2zh_model,en2zh_tok]

    #load English to Japanese model
    mname = 'Helsinki-NLP/opus-mt-en-jap'
    pr_gray("\tload English to Japanese model")
    en2jap_model = MarianMTModel.from_pretrained(mname)
    en2jap_tok = MarianTokenizer.from_pretrained(mname)
    response['en2jap']=[en2jap_model,en2jap_tok]

    if pivot_level == 0 or pivot_level==1:#one pivot language

      #load english to russian model
      mname = 'Helsinki-NLP/opus-mt-en-ru'
      pr_gray("\tload English to Russian model")
      en2ru_model = MarianMTModel.from_pretrained(mname)
      en2ru_tok = MarianTokenizer.from_pretrained(mname)
      response['en2ru']=[en2ru_model,en2ru_tok]

      #load English to German model
      mname = 'Helsinki-NLP/opus-mt-en-de'
      pr_gray("\tload English to German model")
      en2de_model = MarianMTModel.from_pretrained(mname)
      en2de_tok = MarianTokenizer.from_pretrained(mname)
      response['en2de']=[en2de_model,en2de_tok]

      #load Arabic to English model
      mname = 'Helsinki-NLP/opus-mt-ar-en'
      pr_gray("\tload Arabic to English model")
      ar2en_model = MarianMTModel.from_pretrained(mname)
      ar2en_tok = MarianTokenizer.from_pretrained(mname)
      response['ar2en']=[ar2en_model,ar2en_tok]
      
      #load Chinese to English model
      mname = 'Helsinki-NLP/opus-mt-zh-en'
      pr_gray("\tload Chinese to English model")
      zh2en_model = MarianMTModel.from_pretrained(mname)
      zh2en_tok = MarianTokenizer.from_pretrained(mname)
      response['zh2en']=[zh2en_model,zh2en_tok]

      #load Japanese to English model
      mname = 'Helsinki-NLP/opus-mt-jap-en'
      pr_gray("\tload Japanese to English model")
      jap2en_model = MarianMTModel.from_pretrained(mname)
      jap2en_tok = MarianTokenizer.from_pretrained(mname)
      response['jap2en']=[jap2en_model,jap2en_tok]
    
    if pivot_level==0 or pivot_level==2:#Two-pivot language
      #load Chinese to German model
      mname = 'Helsinki-NLP/opus-mt-zh-de'
      pr_gray("\tload Chinese to German model")
      zh2de_model = MarianMTModel.from_pretrained(mname)
      zh2de_tok = MarianTokenizer.from_pretrained(mname)
      response['zh2de']=[zh2de_model,zh2de_tok]

      #load Spanish to Russian model
      mname = 'Helsinki-NLP/opus-mt-es-ru'
      pr_gray("\tload Spanish to Russian model")
      es2ru_model = MarianMTModel.from_pretrained(mname)
      es2ru_tok = MarianTokenizer.from_pretrained(mname)
      response['es2ru']=[es2ru_model,es2ru_tok]

      #load Arabic to German model
      mname = 'Helsinki-NLP/opus-mt-ar-de'
      pr_gray("\tload Arabic to German model")
      ar2de_model = MarianMTModel.from_pretrained(mname)
      ar2de_tok = MarianTokenizer.from_pretrained(mname)
      response['ar2de']=[ar2de_model,ar2de_tok]

      #load Japanes to Spanish model
      mname = 'Helsinki-NLP/opus-mt-ja-es'
      pr_gray("\tload Japanese to Spanish model")
      ja2es_model = MarianMTModel.from_pretrained(mname)
      ja2es_tok = MarianTokenizer.from_pretrained(mname)
      response['ja2es']=[ja2es_model,ja2es_tok]

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
  #load all the model
  print("load model")
  model_list = load_model(2)
  print(len(model_list))
  import sys
  sys.exit()
  main(model_list)
  dataset = ['How does COVID-19 spread?','Book a flight from lyon to sydney?','Reserve a Restaurant at Paris']
  print("start translation")
  paraphrases = translate_list(dataset,model_list,1)
  print("Result:")
  for key,value in paraphrases.items():
    print(key)
    for i in value:
      print("\t",i)