import torch

""" This code translate sentence using pretrained Open Neural Machine Translator Pretrained Model """

def translate(utterance,model):
    """
    Translate a sentence
    :param utterance: sentence to translate
    :param model: openNMT language paris model, en-de or de-en.
    :return Translated utterance 
    """
    #result = model.translate(utterance)
    return model.translate(utterance)

def load_model(model):
    """
    Load Open-NMT language pair model
    :param model: string id representng the model to download
    :return Open-NMT model
    """
    if model == "en-de":
        en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
        return en2de
    elif model == "de-en":
        de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
        return de2en
    else:
        return None


def test():
    #Load Open-NMT translation model
    print("\nLoad Open-NMT translation model:")
    print("\tload English to German model")
    en2de = load_model("en-de")

    print("\tload German to English model")
    de2en = load_model("de-en")

    #strat translation
    tmp = "sentence to translate to german"
    tmp2 = translate(tmp,en2de)

    #translate back to english
    tmp3 = translate(tmp2,de2en)

    print("German: ",tmp2," - ENglish: ",tmp3)

if __name__ == "__main__":
    # Round-trip translations between English and German:
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    paraphrase = en2de.translate('How does COVID-19 spread?')
    print(paraphrase)
    a = de2en.translate(paraphrase)
    print(a)