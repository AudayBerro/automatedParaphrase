import sys
import spacy
from pprint import pprint
from spacy.tokens.token import Token
from nltk.corpus import wordnet as wn
from six.moves import xrange
import random

nlp = spacy.load('en')

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

    return generated_sentences

def synonym_paraphrase(s):
    return synonym_model(s)

if __name__ == '__main__':
    #x = synonym_model('I am discussing my homework with the teacher.')
    #x = synonym_model('the rabbit quickly ran down the hole')
    #x = synonym_model('John tried to fix his computer by hacking away at it.')
    x = synonym_model('team based multiplayer online first person shooter video game')
    print(x)
