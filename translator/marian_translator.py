from transformers import MarianMTModel,MarianTokenizer

""" This code translate sentence using Huggingface Marian Machine Translation Pretrained Model """

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

    # print(result[0])
    # print(type(result[0]))
    # import sys
    # sys.exit()
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
    print("translate en-it: ",tmp)
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])
    print("translate it-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    print("translate en-es: ",tmp)
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])
    print("translate es-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="fr")
    print("translate en-fr: ",tmp)
    tmp = translate(tmp,model['romance2en'][0],model['romance2en'][1])
    print("translate fr-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2ru'][0],model['en2ru'][1])
    print("translate en-ru: ",tmp)
    tmp = translate(tmp,model['ru2en'][0],model['ru2en'][1])
    print("translate es-en: ",tmp)
    response.add(tmp)
    
  if pivot_level == 0 or pivot_level == 2:
    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="es")
    print("translate en-es: ",tmp)
    tmp = translate(utterance,model['es2ru'][0],model['es2ru'][1])
    print("translate es-ru: ",tmp)
    tmp = translate(utterance,model['ru2en'][0],model['ru2en'][1])
    print("translate ru-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2romance'][0],model['en2romance'][1],trg="fr")
    print("translate en-fr: ",tmp)
    tmp = translate(utterance,model['romance2en'][0],model['romance2en'][1])
    print("translate fr-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,model['en2ru'][0],model['en2ru'][1])
    print("translate en-ru: ",tmp)
    tmp = translate(utterance,model['ru2en'][0],model['ru2en'][1])
    print("translate ru-en: ",tmp)
    response.add(tmp)
  return response

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

    return response

if __name__ == "__main__":
    #load all the model
    print("load model")
    model_list = load_model()

    text = "How does covid-19 spread?"
    # translate(text,'fr','en',model,tok)
    print("start translation")
    paraphrases = multi_translate(text,model_list)
    print("Result:")
    for i in paraphrases:
      print(i)