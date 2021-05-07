import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

"""" This code generate paraphrases using Huggingface T5 model's """

def pr_gray(msg):
    """ Pring msg in gray color font"""
    print("\033[7m{}\033[00m" .format(msg))

def pr_green(msg):
    """ Pring msg in green color font"""
    print("\033[92m{}\033[00m" .format(msg))

def set_seed(seed):
    """ Set the seed for generating random numbers for REPRODUCIBILITY """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(model_name):
    """
    Load Fine-Tuned HuggingFace T5 model
    :param model_name: name of the HuggingFace T5 model to load
    :return HuggingFace T5 model
    """
    model = T5ForConditionalGeneration.from_pretrained(model_name) #auday/paraphraser_model1   auday/paraphraser_model2
    return model

def load_tokenizer(model_name='t5-base'):
    """
    Load HuggingFace T5 Tokenizer
    :param model_name: name of the HuggingFace T5 tokenizer to load
    :return HuggingFace T5 Tokenier
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)#default 't5-base'
    return tokenizer

def check_device():
    """
    Check the availability of NVIDIA GPU, to run the code on GPU instead of CPU
    :return cuda if a NVIDIA GPU is installed on the system, otherwise cpu
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def convert_to_t5_format(sentence):
    """
    Convert sentence to T5 Input format: "paraphrase:"+sentence+"</s>"
    :param sentence: sentence to convert to T5 format
    :return a T5 compatible sentence: "paraphrase:"+sentence+"</s>" 
    """
    sentence =  "paraphrase: " + sentence + " </s>"
    return sentence

def encode_input(tokenizer,text):
    """
    Encode text using T5 Tokenizer by adding special tokens using the tokenizer
    :param tokenizer: T5 Tokenizer
    :param text: sentence to encode
    :return the encoded sentence
    """
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    return encoding

def generate_paraphrase(model,input_ids,attention_masks,max_len=256,num_seq=10):
    """
    Generate Parpahrases using T5 model
    :param model: Huggingface T5 model
    :param input_ids: Indices of input sequence tokens in the T5 vocabulary
    :param attention_masks: Mask to avoid performing attention on padding token indices
    :param max_len: The maximum length of the sequence to be generated
    :param num_seq: int, optional, defaults to 1 – The number of independently computed returned sequences for each element in the batch. Higher value return more sentences
    return list of generated paraphrases in a form of torch.FloatTensor or dictionary if config.return_dict=True(in this code is set to False)
    """
    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True, #  Whether or not to use sampling ; use greedy decoding otherwise.
        max_length= max_len, # The maximum length of the sequence to be generated
        top_k=120, # (int, optional, defaults to 50)– The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p=0.98, # (float, optional, defaults to 1.0)– If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        early_stopping=True, # Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
        num_return_sequences=num_seq
    )
    return beam_outputs

def extract_paraphrases(beam_outputs,tokenizer,utterance):
    """
    This funciton extract paraphrases from beam_outputs# (int, optional, defaults to 1) – The number of independently computed returned se# (int, optional, defaults to 1) – The number of independently computed returned sequences for each element in the batch.quences for each element in the batch.
    :param beam_outputs: T5 generated Paraphrases in a form of torch.FloatTensor
    :param tokenizer: T5 Tokenizer
    :param utterance: initial expression to paraphrase
    :return a Python set containing the generated paraphrases 
    """
    final_outputs = set()
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != utterance.lower() and sent not in final_outputs:
            final_outputs.add(sent)
    return final_outputs

def initialisation(model_name,tokenizer,seed):
    """
    This function initialise T5 by loading model and tokenizer
    :param model_name: name of the HuggingFace T5 model to load
    :param tokenizer: name of the HuggingFace T5 tokenizer to load
    :param seed: seed for generating random numbers for REPRODUCIBILITY
    :return T5-model and T5-tokenizer
    """
    #set_seed(42)#set the seed for generating random numbers for REPRODUCIBILITY

    #load pre-trained T5 paraphraser
    pr_gray("\nLoad Huggingface T5 pre-trained paraphraser model:")
    model = load_model(model_name)
    pr_green("... done")

    #load T5 tokenizer
    pr_gray("\nLoad Huggingface T5 Tokenizer model:")
    tokenizer = load_tokenizer()
    pr_green("... done")

    #check GPU availability
    pr_gray("\nCheck GPU availability")
    device = check_device()
    pr_green ("... device: "+str(device))
    model = model.to(device)

    return model,tokenizer,device


def t5_paraphraser(sent,model_name="auday/paraphraser_model2",flag=0,num_seq=10,max_len=256):
    """
    This function generate parpahrases candidates using pretrained Huggingface T5 transformers model
    :param sent: python dictionary, key:initial sentence, value list of paraphrases candidates
    :param model_name: name of the HuggingFace T5 model to load
    :param flag: integer, flag=0 mean the pipeline start with T5 component, otherwise flag=1
    :param num_seq: number of independently computed returned sequences for each element in the batch. Higher value return more sentences
    :param max_len: The maximum length of the sequence to be generated
    :return a Python dictionary containing a list of paraphrases. Key:initial exression, value a list of paraphrases 
    """

    ###############################
    ## T5 initialisation section ##
    ###############################
    model,tokenizer,device = initialisation("auday/paraphraser_model2",'t5-base',40)

    #######################################
    ## T5 paraphrases generation section ##
    #######################################

    result = dict()
    if flag == 0:#the pipeline start with the weak supervision SBSS component
        for k,v in sent.items():
            #convert sentence to T5 format
            text =  convert_to_t5_format(k)

            #encode the sentence
            encoding = encode_input(tokenizer,text)
            input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

            # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
            #generate paraphrases
            beam_outputs = generate_paraphrase(model,input_ids,attention_masks,max_len)

            #extract(decode) the generated paraphrases
            paraphrases = extract_paraphrases(beam_outputs,tokenizer,k)
            result[k] = list(paraphrases) #convert to list before the insertion

    elif flag == 1:#the pipeline have started with another component(e.g. Pivot-translation, T5, etc)
        for k,v in sent.items():
            #convert sentence to T5 format
            candidates = set()#will contain the generated paraphrases

            #generate paraphrases for the initial expression k
            #convert sentence to T5 format
            text =  convert_to_t5_format(k)

            #encode the sentence
            encoding = encode_input(tokenizer,text)
            input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

            # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
            #generate paraphrases
            beam_outputs = generate_paraphrase(model,input_ids,attention_masks,max_len)

            #extract(decode) the generated paraphrases
            paraphrases = extract_paraphrases(beam_outputs,tokenizer,k)
            candidates.update(list(paraphrases))#convert to list before the insertion

            #generate paraphrases for each element in the values list
            if v:#check if v not empty
                for element in v:
                    #convert sentence to T5 format
                    candidates = set()#will contain the generated paraphrases

                    #generate paraphrases for the initial expression k
                    #convert sentence to T5 format
                    text =  convert_to_t5_format(element)

                    #encode the sentence
                    encoding = encode_input(tokenizer,text)
                    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

                    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
                    #generate paraphrases
                    beam_outputs = generate_paraphrase(model,input_ids,attention_masks,max_len)

                    #extract(decode) the generated paraphrases
                    paraphrases = extract_paraphrases(beam_outputs,tokenizer,element)
                    candidates.update(list(paraphrases))#convert to list before the insertion
                
                candidates.update(v)#add K list of parpahrases to result to avoid loosing previous parpahrases 
            result[k] = list(candidates)

    return result

def test():
    sentence = {"book a flight from lyon to sydney":[]}
    final_outputs = t5_paraphraser(sentence)

    for k,v in final_outputs.items():
        print("\t{}: {}".format(k, v))

if __name__ == "__main__":
    test()