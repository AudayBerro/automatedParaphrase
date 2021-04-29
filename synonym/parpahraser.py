import spacy
from nltk.corpus import wordnet as wn
from six.moves import xrange
import random
import spacy_universal_sentence_encoder
import contractions
import re

sim_model = spacy_universal_sentence_encoder.load_model('en_use_lg')
nlp = spacy.load('en')


def get_similarity(token,synonym):
    a = sim_model(token)
    b = sim_model(synonym)
    return a.similarity(b)

def expand_contractions(text):
    """ expand shortened words, e.g. don't to do not """
    return contractions.fix(text)

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

def generate_sentence(original_doc, new_tokens):
    new_sentence = ' '.join(new_tokens).replace('_', ' ')
    new_sentence = normalize_text(new_sentence)
    return new_sentence

def join_set(data1,data2):
    """
    This function join two Sets
    :param :data1 first set
    :param :data2 second set
    :return the union of the two sets
    """
    return set(data1).union(set(data2))

def synonym_model(s,tags):
    """
    This algorithm is based on synonym_model.py module form  Tili-dev/paraphraser  github project: https://github.com/Tili-dev/paraphraser/blob/master/paraphraser/synonym_model.py
    :param s: sentence to generate paraphrases for
    :param tags: par-of-speech tag to focus on during generation
    :return a set of generated parphrases
    """
    generated_sentences = set([])

    doc = nlp(s)
    original_tokens = [ token.text for token in doc ]

    index_to_lemmas = {}

    for index, token in enumerate(doc):
        index_to_lemmas[index] = set([])
        index_to_lemmas[index].add(token)

        if token.pos_ in tags and len(token.text) >= 3:
            if token.pos_ == 'NOUN':
                pos = wn.NOUN
            elif token.pos_ == 'VERB':
                pos = wn.VERB
            #uncomment the following condition to add adjectif and adverb
            # elif token.pos_ == 'ADV' and len(token.text) >= 3:
            #     pos = wn.ADV
            # elif token.pos_ == 'ADJ' and len(token.text) >= 3:
            #     pos = wn.ADJ
            else:
                continue
        else:
            continue

        # Synsets
        for synset in wn.synsets(token.text, pos):
            for lemma in synset.lemmas():
                lemma_token_similarity = get_similarity(lemma.name(),token.text)
                if lemma_token_similarity > 0.5: # if the candidate synonym and token are related e.g. for the sentence "How COVID spread", spread and coif are not related
                    new_tokens = original_tokens.copy()
                    new_tokens[index] = lemma.name().lower()

                    # sentence_and_score = generate_sentence(doc, new_tokens)
                    # generated_sentences.add(sentence_and_score)
                    
                    new_sentence = generate_sentence(doc, new_tokens)
                    generated_sentences.add(new_sentence)

                    index_to_lemmas[index].add(lemma.name())

    count = sum([ len(words) for words in index_to_lemmas.values() ])

    for i in xrange(min(count, 40)):
        new_tokens = []
        for index, words in sorted(index_to_lemmas.items(), key=lambda x: x[0]):
            token = random.sample(index_to_lemmas[index], 1)[0]
            new_tokens.append(str(token))
        sentence_and_score = generate_sentence(doc, new_tokens)
        generated_sentences.add(sentence_and_score)

    #print(generated_sentences)
    return generated_sentences

def get_paraphrases_list(sentence):
    """
    This functionon generate a list of sentence's parpahrases by replacing token with synomnym
    :parma sentence: sentence to generate parpahrases for
    :return Python set of sentence's paraphrases
    """
    # generate parpahrases by replacing NOUN with synonym
    tags = ['NOUN'] # if token.pos in tags replace get synonym
    data1 = synonym_model(sentence,tags)
    data1 = sorted(data1, key = lambda x: x[1], reverse = True)

    # generate parpahrases by replacing VERB with synonym
    tags = ['VERB']
    data2 = synonym_model(sentence,tags)
    data2 = sorted(data2, key = lambda x: x[1], reverse = True)

    # generate parpahrases by replacing NOUN and VERB with synonym
    tags = ['NOUN','VERB']
    data3 = synonym_model(sentence,tags)
    data3 = sorted(data3, key = lambda x: x[1], reverse = True)

    # generate parpahrases by replacing NOUN, VERB, ADVERBE and ADJECTIVE with synonym
    # tags = ['NOUN','VERB','ADV','ADJ']
    # data4 = synonym_model(sentence,tags)
    # data4 = sorted(data4, key = lambda x: x[1], reverse = True)

    result = join_set(data1,data2)
    result = join_set(result,data3)
    return result

def main(file_path):
    f=open(file_path, "r")
    result = dict()
    while True: 
        # Get next line from file 
        line = f.readline()
        if not line: 
            break

        sent = expand_contractions(line)  #expand contraction e.g can't -> can not
        sent = normalize_text(sent)
        paraphrases = get_paraphrases_list(sent)
        line = line.replace('\n', '').replace('\r', '')
        result[line] = paraphrases
    
    return result


def gui_main(sentence):
    """
    Apply Weak Supervision to generate paraphrases candidate using (pipeline SRSS component)
    :param sentence: sentence to generate parpahrases for
    :return a list of generated paraphrases using the SRSS part of the Weak-supervision component of the pipeline
    """
    sent = expand_contractions(sentence)  #expand contraction e.g can't -> can not
    sent = normalize_text(sent)
    paraphrases = get_paraphrases_list(sent)

    #convert paraphrases to python list (paraphrases is a python set)
    return list(paraphrases) 

if __name__ == '__main__':
    #import sys
    #import os
    #file_path = os.path.join(os.path.dirname(__file__), "..", "dataset/"+sys.argv[1])
    # pool = main(file_path)
    # print(pool)
    pool = gui_main('book a flight from lyon to sydney')
    print(pool)

