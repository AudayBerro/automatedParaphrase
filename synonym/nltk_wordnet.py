from nltk.corpus import wordnet as wm
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import wmd

""" Get token synonym using NLTK wordnet Corpus """

def wordnet_spacy():
    # Load an spacy model (supported models are "es" and "en") 
    nlp = spacy.load('en')
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
    token = nlp('prices')[0]

    # wordnet object link spacy token with nltk wordnet interface by giving acces to
    # synsets and lemmas 
    token._.wordnet.synsets()
    token._.wordnet.lemmas()

    # And automatically tags with wordnet domains
    token._.wordnet.wordnet_domains()

    # Imagine we want to enrich the following sentence with synonyms
    sentence = nlp('I want to withdraw 5,000 euros')

    # spaCy WordNet lets you find synonyms by domain of interest
    # for example economy
    economy_domains = ['finance']
    enriched_sentence = []

    # For each token in the sentence
    for token in sentence:
        # We get those synsets within the desired domains
        synsets = token._.wordnet.wordnet_synsets_for_domain(economy_domains)
        if synsets:
            lemmas_for_synset = []
            for s in synsets:
                # If we found a synset in the economy domains
                # we get the variants and add them to the enriched sentence
                lemmas_for_synset.extend(s.lemma_names())
                enriched_sentence.append('({})'.format('|'.join(set(lemmas_for_synset))))
        else:
            enriched_sentence.append(token.text)

    # Let's see our enriched sentence
    print(' '.join(enriched_sentence))

def ignore_stop_words(sentence):
    """
    Replace stop words by a mask token
    :param sentence: sentence 
    :return a string where stop words are replaced with XXXSTOPXXX Mask and a list containing the replaced stop words
    """
    tknzr = TweetTokenizer()
    sent = tknzr.tokenize(sentence)#tokenize the sentence
    print("## : ",sent)
    # if word in stopwords.words():
    #    pass
def get_synonym(word,pos_tag):
    """
    Get List of sentence synonym
    :param word: get a list of word synonym
    :param pos_tag: part-of-speech tag, only extract synonym that have the same part-of-speech tag from wordnet e.g: dog synset[Synset('dog.n.01'),Synset('chase.v.01')]; n=noun - v=verb
    :return a list of synonym that have the same part-of-speech tag with word
    """

    syn = wm.synsets(word)

    syn_list = set()
    for token in syn:
        #print("\n\t",token.definition()," - ",token.name()," - ",token.lemmas()," - ",token.lemmas()[0].name())
        print(token.examples()[0])
        syn_list.add(token.lemmas()[0].name()) # or token.lemma_names() will get list of token lemmas ex: ['fail', 'go_bad', 'give_way', 'die','break_down']
    
    for i in syn_list:
        print(i)
    
    return syn_list

def get_best_synonym(word,sentence,synonyms):
    """
    Select appropriate synonym to word from synonyms list based on sentence
    :param 
    :param sentence: sentence 
    :param synonyms: list of synonym
    :return a sentence where word is replaced by the best synonym
    """

    nlp = spacy.load('en_core_web_lg', create_pipeline=wmd.WMD.create_spacy_pipeline)
    sent1 = nlp(sentence)
    max_score = 0
    for candidate in synonyms:
        sent2 = sentence.replace(word,candidate)
        sent2 = nlp(sent2)
        score = sent1.similarity(sent2)
        print("\n\n",sent1," - ",sent2)
        print("candidate: ",candidate," - score: ",score)
        if score > max_score:
            result = candidate
            max_score =score
    
    return candidate

def main(word,pos_tag):
    """
    Get List of sentence synonym
    :param word: get a list of word synonym
    :param pos_tag: part-of-speech tag, only extract synonym that have the same part-of-speech tag from wordnet e.g: dog synset[Synset('dog.n.01'),Synset('chase.v.01')]; n=noun - v=verb
    :return a list of synonym that have the same part-of-speech tag with word
    """
    # nltk.download() #import nltk  run this command at the first time to download the wordnet corpus
    syn = wm.synsets(word)
    # print(len(syn))

    # #synset
    # print(syn)

    # #synset definition
    # print(syn[0].definition())
    
    syn_list = set()
    for token in syn:
        #print("\n\t",token.definition()," - ",token.name()," - ",token.lemmas()," - ",token.lemmas()[0].name())
        # print(token.examples()[0])
        a = str(token.lemmas()[0]).split('.')#lemmas is of form (synonym,Part-of-speech,meaning)
        # print(a[0]," - ",a[1]," - ",a[2])
        if a[1] in pos_tag:
            sent = a[0].replace("Lemma('","")# remove #Lemma(' form the string: Lemma('spread
            if sent != word:
                if '_' in sent:
                    sent = sent.replace('_'," ") 
                syn_list.add(sent)
        # syn_list.add(token.lemmas()[0].name()) # or token.lemma_names() will get list of token lemmas ex: ['fail', 'go_bad', 'give_way', 'die','break_down']
    
    return list(syn_list)

if __name__ == "__main__":
    # wordnet_spacy()
    import sys
    sys.path.append("..")
    from pos import pos_extraction as ps
    # sent = ps.pos_extraction2("../dataset/dataset.txt",['VERB','NOUN']) # get pos tags
    tags = ['VERB','NOUN'] #list of tag to extract from sentence using spacy

    f=open("../dataset/dataset.txt", "r")
    s = []
    while True: 
        # Get next line from file 
        line = f.readline()
        if not line: 
            break
        line = line.rstrip('\n')
        print("Line: ",line)
        line = ps.expand_contractions(line)  #expand contraction e.g can't -> can not

        Candidate,tokenized_list = ps.sentence_pos(line,tags)
        
        result = []

        for token in tokenized_list:
            if token in Candidate:
                wordnet_synonym = main(token,['v','n'])
                print(">",token," - ",wordnet_synonym)

                if wordnet_synonym:
                    best_synonym = get_best_synonym(token,line,wordnet_synonym) #get best synonym
                    result.append(best_synonym)
                else:
                    result.append(token)
            else:
                result.append(token)
        
        result = " ".join(result)
        s.append(line+" - "+result)
    
    
    print("\n\n===========================================================")
    for i in s:
        print(i)