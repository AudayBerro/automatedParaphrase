import spacy
import contractions
from spacy.lang.en.stop_words import STOP_WORDS

""" Part of Speech extraction using Spacy library """


def pos_extraction(file_name):
  """
  Part of Speech extraction
  :param file_name: file name to access data on which to apply POS extraction
  :return a Python dictionary, Key is the initial expression and value is a pos_extracted sentence
  """
  
  nlp = spacy.load('en_core_web_lg') # en_core_web_sm

  pos_list = ['NOUN','VERB','ADV','ADJ','PRON'] # element to extract from each sentence

  f=open(file_name, "r")
  response = dict()
  while True:
    # Get next line from file 
    line = f.readline()
    if not line: 
      break

    sentences = nlp(line)
    para = [] 
    for token in sentences:
      if token.text == '\n':#to remove breakline at the end of sentence
        continue
      if token.pos_ in pos_list:#Part of speech tagging to extract NOUN,VERB,ADV,ADJ
        para.append(token.text)
    value = " ".join(para)
    response[line.rstrip('\n')] = value
  
  return response

def pos_extraction2(file_name,tags):
  """
  Specific Part of Speech extraction to generate new data corpus (apply Weak supervision approach)
  :param file_name: file name to access data on which to apply POS extraction
  :param tags: list of tag to extract for each sentence
  :return a list containing the new generated data
  """
  
  nlp = spacy.load('en_core_web_lg') # en_core_web_lg

  f=open(file_name, "r")
  response = []
  while True:
    # Get next line from file 
    line = f.readline()
    if not line: 
      break

    sentences = nlp(line)
    para = [] 
    for token in sentences:
      if token.text == '\n':
        continue
      if token.pos_ in tags:
        para.append(token.text)
    value = " ".join(para)
    response.append(value)
  
  return response

def expand_contractions(text):
    """ expand shortened words, e.g. don't to do not """

    #pycontraction library
    # Choose model accordingly for contractions function
    # model = api.load("glove-twitter-25")
    # # model = api.load("glove-twitter-100")
    # # model = api.load("word2vec-google-news-300")cont = Contractions(kv_model=model)
    # cont.load_models()
    # text = list(cont.expand_texts([text], precise=True))[0]

    result = contractions.fix(text)
    return result

def sentence_pos(sentence,tags):
  """
  Specific Part of Speech extraction to generate new data corpus (apply Weak supervision approach)
  :param sentence: sentence
  :param tags: list of tag to extract for each sentence
  :return a list containg word that have the same POS tags defined by tags and the sentence in a form of list after tokenization
  """
  
  nlp = spacy.load('en_core_web_lg') # en_core_web_lg

  # sentence = expand_contractions(sentence) #expand contraction e.g can't -> can not

  sentences = nlp(sentence)
  candidate = []
  tokenized_sentence = []
  
  for token in sentences:
    if token.text == '\n':
      continue
    tokenized_sentence.append(token.text) #add token
    if token.pos_ in tags and token.text not in STOP_WORDS:
      candidate.append(token.text)
  
  return candidate,tokenized_sentence