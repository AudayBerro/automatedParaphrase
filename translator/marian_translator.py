from transformers import MarianMTModel,MarianTokenizer
import os
import contractions
import re,string

""" This code translate sentence using Huggingface Marian Machine Translation Pretrained Model """

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
    translated = model.generate(**tok.prepare_translation_batch([utterance]))
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
  if pivot_level == 0 or pivot_level == 1:  
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="it")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])
    tmp = expand_contractions(tmp)
    tmp = normalize_text(tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])
    tmp = expand_contractions(tmp)
    tmp = normalize_text(tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="fr")
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])
    tmp = expand_contractions(tmp)
    tmp = normalize_text(tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2ru'][0],model['en2ru'][1])
    tmp = translate(tmp,model['ru2en'][0],model['ru2en'][1])
    tmp = expand_contractions(tmp)
    tmp = normalize_text(tmp)
    response.add(tmp)
    
  if pivot_level == 0 or pivot_level == 2:
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    tmp = translate(utterance,model['es2ru'][0],model['es2ru'][1])
    tmp = translate(utterance,model['ru2en'][0],model['ru2en'][1])
    tmp = expand_contractions(tmp)
    tmp = normalize_text(tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="fr")
    tmp = translate(utterance,model['fr2ru'][0],model['fr2ru'][1])
    tmp = translate(utterance,model['ru2en'][0],model['ru2en'][1])
    tmp = expand_contractions(tmp)
    tmp = normalize_text(tmp)
    response.add(tmp)
  return response

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

def load_model():
    """
    Return a List of Huggingface Marian MT model
    :return Python dictionary - key: model name - value: list containing respectively MarianModel and MarianTokenizer e.g. {'en2ru':[model,tokenizer]}
    """
    response = dict()
    #load model to translate from en to ['French','Spanish','Italian','Portuguese']
    print("load English to Romance model")
    mname1 = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    en2romance_model = MarianMTModel.from_pretrained(mname1) #load model
    en2romance_tok = MarianTokenizer.from_pretrained(mname1) #load tokenizer
    response['en2romance']=[en2romance_model,en2romance_tok]

    #load model to translate from ['French','Spanish','Italian','Portuguese'] to english
    print("load Romance to English model")
    mname2 = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    romance_en_model = MarianMTModel.from_pretrained(mname2) #load model
    romance_en_tok = MarianTokenizer.from_pretrained(mname2) #load tokenizer
    response['romance2en']=[romance_en_model,romance_en_tok]

    #load english to russian model
    mname = 'Helsinki-NLP/opus-mt-en-ru'
    print("load English to Russian model")
    en2ru_model = MarianMTModel.from_pretrained(mname)
    en2ru_tok = MarianTokenizer.from_pretrained(mname)
    response['en2ru']=[en2ru_model,en2ru_tok]

    #load russian to english model
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    print("load Russian to English model")
    ru2en_model = MarianMTModel.from_pretrained(mname)
    ru2en_tok = MarianTokenizer.from_pretrained(mname)
    response['ru2en']=[ru2en_model,ru2en_tok]

    #load Spanish to Russian model
    mname = 'Helsinki-NLP/opus-mt-es-ru'
    print("load Spanish to Russian model")
    es2ru_model = MarianMTModel.from_pretrained(mname)
    es2ru_tok = MarianTokenizer.from_pretrained(mname)
    response['es2ru']=[es2ru_model,es2ru_tok]

    #load French to Russian model
    mname = 'Helsinki-NLP/opus-mt-fr-ru'
    print("load French to Russian model")
    fr2ru_model = MarianMTModel.from_pretrained(mname)
    fr2ru_tok = MarianTokenizer.from_pretrained(mname)
    response['fr2ru']=[fr2ru_model,fr2ru_tok]

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
  model_list = load_model()
  main(model_list)
  dataset = ['How does COVID-19 spread?','Book a flight from lyon to sydney?','Reserve a Restaurant at Paris']
  print("start translation")
  paraphrases = translate_list(dataset,model_list,1)
  print("Result:")
  for key,value in paraphrases.items():
    print(key)
    for i in value:
      print("\t",i)