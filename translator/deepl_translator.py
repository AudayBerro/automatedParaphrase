import requests

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
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'RU',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'FR',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'JA',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'DE',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)
    
  if pivot_level == 0 or pivot_level == 2:
    tmp = translate(text,'IT',api_key)
    tmp = translate(tmp,'RU',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'IT',api_key)
    tmp = translate(tmp,'DE',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'RU',api_key)
    tmp = translate(tmp,'FR',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'RU',api_key)
    tmp = translate(tmp,'JA',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'JA',api_key)
    tmp = translate(tmp,'FR',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)

    tmp = translate(text,'JA',api_key)
    tmp = translate(tmp,'DE',api_key)
    response.add(google.translate_wrapper(tmp,'en'))
    tmp = translate(tmp,'EN',api_key)
    response.add(tmp)
  return response

if __name__ == "__main__":
    print(translate('je mange.tu mange."ferme la"','EN','f55c628f-a052-4431-90a7-86d0b8ca861b'))
    
