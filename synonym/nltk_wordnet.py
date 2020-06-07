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
        sent2 = sentence.replace(word,candidate) #replace word by candidate synonym
        sent2 = nlp(sent2)
        score = sent1.similarity(sent2) #compute word mover distance to select the best synonym, we can also use cosine similarity on BERT embedding
        if score > max_score:
            result = candidate
            max_score =score
    
    return result

def get_synonym(word,pos_tag):
    """
    Get List of sentence synonym
    :param word: get a list of word synonym
    :param pos_tag: part-of-speech tag, only extract synonym that have the same part-of-speech tag from wordnet e.g: dog synset[Synset('dog.n.01'),Synset('chase.v.01')]; n=noun - v=verb
    :return a list of synonym that have the same part-of-speech tag with word
    """
    # nltk.download() #import nltk  run this command at the first time to download the wordnet corpus
    syn = wm.synsets(word)
    
    syn_list = set()
    for token in syn:
        a = str(token.lemmas()[0]).split('.')#lemmas is of form (synonym,Part-of-speech,meaning)
        if a[1] in pos_tag:
            sent = a[0].replace("Lemma('","")# remove #Lemma(' form the string: Lemma('spread
            if sent != word:
                if '_' in sent: #remove underscore if exist e.g: go_arround -> go arround
                    sent = sent.replace('_'," ") 
                syn_list.add(sent)
    
    return list(syn_list)

def main(file_path,pos_tags,wordnet_tags):
    """
    Apply part-of-speech taging, replace detected token defined in tags by appropriate wordnet synonym
    :param file_path: file path
    :param pos_tags: select only token which pos-tags is in pos_tags as candidate token to replace with wordnet synonym
    :param pos_tags: select wordnet synset lemmas which pos-tags is in wordnet_tags
    :return a new dataset where words selected from a list of tags are replaced by a synonymous word.
    """
    # wordnet_spacy()
    import sys
    sys.path.append("..")
    from pos import pos_extraction as ps
    
    f=open(file_path, "r")
    result = []
    while True: 
        # Get next line from file 
        line = f.readline()
        if not line: 
            break
        line = line.rstrip('\n')
        line = ps.expand_contractions(line)  #expand contraction e.g can't -> can not

        Candidate,tokenized_list = ps.sentence_pos(line,pos_tags)
        sentence = []

        for token in tokenized_list:
            if token in Candidate:
                wordnet_synonym = get_synonym(token,wordnet_tags)

                if wordnet_synonym:
                    best_synonym = get_best_synonym(token,line,wordnet_synonym) #get best synonym
                    sentence.append(best_synonym)
                else:
                    sentence.append(token)
            else:
                sentence.append(token)
        
        sentence = " ".join(sentence)
        # s.append(line+" - "+sentence)
        result.append(sentence)
    
    return result