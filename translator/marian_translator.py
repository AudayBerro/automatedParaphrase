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


def multi_translate(utterance,pivot_level=1):
  """
  Translate sentence
  :param utterance: sentence to translate
  :param pivot_level: integer that indicate the pivot language level, single-pivot or multi-pivot range,1 =single-pivot, 2=double-pivot, 0=apply single and double
  :return list of utterance translations
  """
  #'>>fr<< Comment la COVID-19 se propage-t-elle?',
  #     '>>pt<< Isto deve ir para o português.',
  #     '>>es<< Y esto al español'

  #en-ROMANCE supported language: #['>>fr<<', '>>es<<', '>>it<<', '>>pt<<']
  response = set()
  if pivot_level == 0 or pivot_level == 1:
    #load model to translate from en to ['French','Spanish','Italian','Portuguese']
    mname1 = f'Helsinki-NLP/opus-mt-en-ROMANCE'
    en_romance_model = MarianMTModel.from_pretrained(mname1) #load model
    en_romance_tok = MarianTokenizer.from_pretrained(mname1) #load tokenizer

    #load model to translate from ['French','Spanish','Italian','Portuguese'] to english
    mname2 = f'Helsinki-NLP/opus-mt-ROMANCE-en'
    romance_en_model = MarianMTModel.from_pretrained(mname2) #load model
    romance_en_tok = MarianTokenizer.from_pretrained(mname2) #load tokenizer
    
    tmp = translate(utterance,en_romance_model,en_romance_tok,trg="it")
    print("trnaslate en-it: ",tmp)
    tmp = translate(utterance,romance_en_model,romance_en_tok)
    print("trnaslate it-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,en_romance_model,en_romance_tok,trg="es")
    print("trnaslate en-es: ",tmp)
    tmp = translate(utterance,romance_en_model,romance_en_tok)
    print("trnaslate es-en: ",tmp)
    response.add(tmp)

    tmp = translate(utterance,en_romance_model,en_romance_tok,trg="fr")
    print("trnaslate en-fr: ",tmp)
    tmp = translate(utterance,romance_en_model,romance_en_tok)
    print("trnaslate fr-en: ",tmp)
    response.add(tmp)

    src = 'en'  # source language
    trg = 'ru'  # target language
    #load english to russian model
    mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    en_ru_model = MarianMTModel.from_pretrained(mname)
    en_ru_tok = MarianTokenizer.from_pretrained(mname)
    tmp = translate(utterance,en_ru_model,en_ru_tok)
    print("trnaslate en-ru: ",tmp)
    src = 'en'  # source language
    trg = 'ru'  # target language
    #load russian to english model
    mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    en_ru_model = MarianMTModel.from_pretrained(mname)
    en_ru_tok = MarianTokenizer.from_pretrained(mname)
    tmp = translate(utterance,romance_en_model,romance_en_tok)
    print("trnaslate es-en: ",tmp)
    response.add(tmp)
    
  if pivot_level == 0 or pivot_level == 2:
    print("NOTHING GULCH")
  return response

if __name__ == "__main__":
    # mname = f'Helsinki-NLP/opus-mt-fr-en'
    # model = MarianMTModel.from_pretrained(mname)
    # tok = MarianTokenizer.from_pretrained(mname)
    text = "How does covid-19 spread?"
    # translate(text,'fr','en',model,tok)
    paraphrases = multi_translate(text)
    print(len(paraphrases))
    print("\n\n================")
    for i in paraphrases:
      print(i)