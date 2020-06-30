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


def translate(utterance,src,trg,model,tok):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param src: source language
    :param trg: target language
    :param model: transformers Marian Machine Transaltion Model module(MarianMTModel)
    :param tok: transformers Marian Tokenizer module(MarianTokenizer)
    :return Translated utterance 
    """
    batch = tok.prepare_translation_batch(src_texts=[utterance])
    gen = model.generate(**batch)
    words = tok.batch_decode(gen, skip_special_tokens=True)

    print(words)

if __name__ == "__main__":
    mname = f'Helsinki-NLP/opus-mt-fr-en'
    model = MarianMTModel.from_pretrained(mname)
    tok = MarianTokenizer.from_pretrained(mname)
    text = "où est l'arrêt de bus ?"
    translate(text,'fr','en',model,tok)