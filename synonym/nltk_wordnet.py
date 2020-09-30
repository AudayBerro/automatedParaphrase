from nltk.corpus import wordnet as wm
from nltk.tokenize import word_tokenize
import wmd

""" Get token synonym using NLTK wordnet Corpus """

def normalize_text(text):
    """
    Remove line break and lowercase all words
    :param text: sentence to normalize
    :return return a sentence without line break and lowercased 
    """
    return text.replace('\n', ' ').replace('\r', '').lower()

def get_best_synonym(word,sentence,synonyms,nlp):
    """
    Select appropriate synonym to word from synonyms list based on sentence (Select the best word synonym)
    :param word: the word to be replaced by a synonym
    :param sentence: sentence 
    :param synonyms: list of synonym
    :param nlp: spacy model
    :return a sentence where word is replaced by the best synonym
    """
    sent1 = nlp(sentence)
    max_score = 0
    for candidate in synonyms:
        sent2 = sentence.replace(word,candidate) #replace word by candidate synonym
        sent2 = nlp(sent2)
        score = sent1.similarity(sent2) #compute word mover distance to select the best synonym, we can also use cosine similarity on BERT embedding
        if score > max_score:
            result = candidate
            max_score = score
    
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

    import sys
    sys.path.append("..")
    from pos import pos_extraction as ps

    # import spacy
    # nlp = spacy.load('en_core_web_lg', create_pipeline=wmd.WMD.create_spacy_pipeline) # load spacy model, add Word Mover Distance pipeline

    #Universal sentence embedding with spacy https://github.com/MartinoMensio/spacy-universal-sentence-encoder
    import spacy_universal_sentence_encoder
    nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
    
    f=open(file_path, "r")
    result = []
    while True: 
        # Get next line from file 
        line = f.readline()
        if not line: 
            break

        line = ps.expand_contractions(line)  #expand contraction e.g can't -> can not
        line = normalize_text(line) #lowercase the sentence help to avoid wordnet word confusion. Wordnet consider Can as the beverage bottle and not the verb

        candidate,tokenized_list = ps.sentence_pos(line,pos_tags)
        sentence = []

        for token in tokenized_list:
            if token in candidate:
                wordnet_synonym = get_synonym(token,wordnet_tags)

                if wordnet_synonym:
                    best_synonym = get_best_synonym(token,line,wordnet_synonym,nlp) #get best synonym
                    sentence.append(best_synonym)
                else:
                    sentence.append(token)
            else:
                sentence.append(token)
        
        sentence = " ".join(sentence)
        result.append(sentence)
    
    return result