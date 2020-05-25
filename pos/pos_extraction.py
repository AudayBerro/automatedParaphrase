import spacy

""" Part of Speech extraction using Spacy library """


def postExtraction(file_name):
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