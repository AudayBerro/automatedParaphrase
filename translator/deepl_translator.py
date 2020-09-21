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

if __name__ == "__main__":
    print(translate('je mange.tu mange."ferme la"','EN','f55c628f-a052-4431-90a7-86d0b8ca861b'))
    
