import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

"""" This code generate paraphrases using Huggingface T5 model's """

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
    model = T5ForConditionalGeneration.from_pretrained(model_name) #auday/t5_paraphraser/model1   auday/t5_paraphraser/model2
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
    return a T5 compatible sentence: "paraphrase:"+sentence+"</s>" 
    """
    sentence =  "paraphrase: " + sentence + " </s>"
    return sentence


def test():
    set_seed(42)

    model = T5ForConditionalGeneration.from_pretrained('./models/model1')#import pre-trained T5 model from ./Transformers/models
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)
    model = model.to(device)

    sentence = "how does covid-19 spread"


    text =  "paraphrase: " + sentence + " </s>"

    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True, #  Whether or not to use sampling ; use greedy decoding otherwise.
        max_length=256, # The maximum length of the sequence to be generated
        top_k=120, # (int, optional, defaults to 50)– The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p=0.98, # (float, optional, defaults to 1.0)– If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        early_stopping=True, # Whether to stop the beam search when at least num_beams sentences are finished per batch or not.
        num_return_sequences=10 # (int, optional, defaults to 1) – The number of independently computed returned sequences for each element in the batch.
    )


    print ("\nOriginal Question ::")
    print (sentence)
    print ("\n")
    print ("Paraphrased Questions :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    for i, final_output in enumerate(final_outputs):
        print("{}: {}".format(i, final_output))

if __name__ == "__main__":
    test()