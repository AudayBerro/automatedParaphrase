
  

# Automated Paraphrasing Tool

We offer a tool to automatically generate paraphrases. The tool first generate paraphrases, remove semantically irrelevant and de-duplicate paraphrases by applying **cosine similarity** on **word embedding vector** of different embedding model as proposed by Parikh[[1]](#1).

## Features

- Automated translation using Online Translator(e.g.[DeepL API](https://www.deepl.com/en/docs-api/) and [MyMemory API](https://mymemory.translated.net/doc/)) or Pretrained Neural Translation Model(e.g.[Huggingface MarianMT](https://huggingface.co/transformers/model_doc/marian.html) and [Open NMT](https://opennmt.net/Models-py/))
- Apply **Weak Supervision Approach**[[2]](#2) to generate more data
- Filter out bad paraphrases through [Hugging Face's transformers BERT model](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) and [Universal Sentence Encoding](https://tfhub.dev/google/universal-sentence-encoder/4) semantic similarity
- Remove deduplicate through [Hugging Face's transformers BERT model](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)

 

Installation & Usage
---------------
In order to generate paraphrases, follow these steps:
  
1. Create and activate a virtual environment using **Python 3** version:

*  `Linux`

   Create the virtual environment: ```virtualenv -p python3 venv ``` 
   Activate the virtual environment: ``` source ./env/bin/activate ```

*  `Windows`

   Create the virtual environment: ``` c:\>c:\Python35\python -m venv c:\path\to\myenv ```

   >Unlike most Unix systems and services, Windows does not include a system supported installation of Python. [#Windows Python installation and Creation of virtual environments](https://docs.python.org/3/using/windows.html#using-on-windows)

    Activate the virtual environment: ``` .\env\Scripts\activate.bat ```

2. Install the required packages inside the environment:

   ``` 
   pip install -r requirements.txt
   ```

3. Download Spacy models, for more models see [Spacy Models & Languages](https://spacy.io/models/en).

   ```
   python -m spacy download en_core_web_lg
   ```

4. Make use of Google's Universal Sentence Encoder directly within SpaCy[Spacy - Universal Sentence Encoder](https://github.com/MartinoMensio/spacy-universal-sentence-encoder).

   ```
   pip install spacy-universal-sentence-encoder
   ```

5. Open **config.ini** configuration file and update the values.

   >  **Note**: Please make sure you fulfilled the required configs in **config.ini** file - especially YANDEX and MYMEMORY.

  

6. Put the sentence you want to paraphrase in a file, sentences should be separated by a line break. Save the file in the **dataset** folder(we suggest to save the file with txt extension).

  

7. Generate paraphrases by runing the following command:
   ```
   $ python main.py -f file_name.txt
   ```

This will save the generated paraphrases in the **result** folder. The **result** file display the paraphrases in Python dictionary(**Key** -- initial sentences and **Value** -- list of paraphrases) following the given order:
>- Paraphrases generated by Translation API,
>- Paraphrases after filtration with [Universal Sentence Encoding](https://tfhub.dev/google/universal-sentence-encoder/4)
>- Paraphrases after filtration with [BERT](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)
>- Paraphrases after duplication with [BERT](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)


## References
<a id="1">[1]</a> Parikh, Soham, Quaizar Vohra, and Mitul Tiwari. "Automated Utterance Generation." _arXiv preprint arXiv:2004.03484_ (2020).

<a id="2">[2]</a> Weir, Nathaniel and Crotty, Andrew and Galakatos, Alex and others. "DBPal: Weak Supervision for Learning a Natural Language Interface to Databases." _arXiv preprint arXiv:1909.06182_ (2019).