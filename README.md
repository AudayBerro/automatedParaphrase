
  

# Automated Paraphrasing Tool

We propose an extensible and reusable pipeline tool that unifies, integrates and extends various paraphrasing techniques(e.g. Weak-Supervision, Pivot-Translation) to automatically generating paraphrases in English that are  semantically  relevant  and  diverse. In doing so, the pipeline describes as two-step process, including:
1. **candidate over-generation**, leveraging techniques that can be combined to generate a large number of diverse but (potentially) noisy candidate paraphrases
2. **candidate selection**, with techniques that can be incorporated to discard semantically irrelevant paraphrases and duplicates, thus filtering out low quality paraphrases.

The pipeline can be run through a command line see `section 6` or by using the pipeline web interface see `section5`.

## Features

- Paraphrases generation through **Pivot-Translation** using Online Machine Translator(e.g.[DeepL API](https://www.deepl.com/en/docs-api/) and [MyMemory API](https://mymemory.translated.net/doc/)) or Pretrained Neural Translation Model(e.g.[Huggingface MarianMT](https://huggingface.co/transformers/model_doc/marian.html) and [EasyNMT](https://github.com/UKPLab/EasyNMT#available-models))
- Paraphrases generation through **Weak Supervision Approach**[[1]](#1) by replacing selected token by relevant [NLTK-WordNet](https://www.nltk.org/howto/wordnet) synomym
- Paraphrases generation using a pretrained [Huggignface T5 Transformer](https://huggingface.co/transformers/model_doc/t5.html).
- Filter out bad paraphrases through [Hugging Face's transformers BERT model](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) and [Universal Sentence Encoding](https://tfhub.dev/google/universal-sentence-encoder/4) semantic similarity
- Remove deduplicate through [Hugging Face's transformers BERT model](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)

 

Virtual Environment Installation
---------------
In order to generate paraphrases, follow these steps:
  
1. Create and activate a virtual environment using **Python 3.6.9** version:

*  `Linux`

   Create the virtual environment:
   ```
   virtualenv -p python3.6.9 my_venv
   ```

   Activate the virtual environment:
   ```
   source ./my_venv/bin/activate
   ```

*  `Windows`

   - Download the desired **Python** version(do NOT add to *PATH*!), and remember the ``` path\to\new_python.exe``` of the newly installed version
   - Create the virtual environment open Command Prompt and enter :
      ```
      virtualenv \path\to\my_env -p path\to\new_python.exe
      ```

   >Unlike most Unix systems and services, Windows does not include a system supported installation of Python. [#Windows Python installation and Creation of virtual environments](https://docs.python.org/3/using/windows.html#using-on-windows)

   - Activate the virtual environment:``` .\my_venv\Scripts\activate.bat ```
   - Deactivate with ``` deactivate ```

Download **Python 3.6.9** - from [linuxize.com](https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/)
---------------
If your working environment does not include the Python=3.6.9 version, apply the following instructions for installation:

1. First, update the packages list and install the packages necessary to build Python source: 
   ```
   $ sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev 
   ```

2. Download the source code from the [Python download page](https://www.python.org/downloads/source/) using the following command:
   ```
   $ wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
   ```

3. Once the download is complete, extract the gzipped tarball:
   ```
   $ tar -xf Python-3.6.9.tgz
   ```

4. Next, navigate to the Python source directory and run the configure script which will perform a number of checks to make sure all of the dependencies on your system are present:
   ```
   $ cd Python-3.6.9
   $ ./configure --enable-optimizations
   ```
   The **--enable-optimizations** option will optimize the Python binary by running multiple tests. This makes the build process slower.

5. Start the Python build process using **make**:
   ```
   $ make -j 8
   ```
   For faster build time, modify the **-j** flag according to your processor. If you do not know the number of cores in your processor, you can find it by typing **nproc**. The system used in this guide has 8 cores, so we are using the **-j8** flag

6. When the build is done, install the Python binaries by running the following command:
   ```
   $ sudo make altinstall
   ```
   Do not use the standard **make install** as it will overwrite the default system python3 binary.

7. That’s it. Python 3.6.9 has been installed and ready to be used. Verify it by typing:
   ```
   $ python3.6.9 --version
   ```

Dependencies modules Installation
---------------
1. Install the required packages inside the environment:

   ``` 
   pip install -r requirements.txt
   ```

2. Download Spacy models, for more models see [Spacy Models & Languages](https://spacy.io/models/en).

   ```
   python -m spacy download en_core_web_lg
   ```

3. Make use of Google's Universal Sentence Encoder directly within [Spacy - Universal Sentence Encoder](https://github.com/MartinoMensio/spacy-universal-sentence-encoder).

   ```
   pip install spacy-universal-sentence-encoder
   ```
   > **Note**: By default we use en_use_lg model, if you want to use another model, modify **load_model** in ./synonym/nltk_wordnet.py line 76 

Usage
---------------

1. Run the pipeline using the web interface
   >- run `app.py` script: ```python app.py```
   >- open any browser and enter the following URL: ```http://localhost:5000/```

2. Run the pipeline using the command line
Open **config.ini** configuration file and update the values.

   >  **Note**: Please make sure you fulfilled the required configs in **config.ini** file - especially DEEPL and MYMEMORY.

    a. Put the sentence you want to paraphrase in a file, sentences should be separated by a line break. Save the file in the **dataset** folder(we suggest to save the file with txt extension).
    
    b. Generate paraphrases by runing the following command:
        ```
        $ python main.py -f dataset.txt -l 1 -p false
        ```

## Command line pipeline Parameter
| Parameter | Description |
| ------ | ------ |
| **-f** | initial data file path, utterances should be separated by a breakline and file extension **.txt** |
| **-p** | generate paraphrases using Online Translator Model(Google,DeepL,Yandex) or pretrained Neural Machine Translator(MariamMT,OpenNMT).<ul><li>**-p true** translate with pretrained Neural Machine Translator asign.</li><li>**-p false** translate with Online Neural Translator  Model.</li></ul>|
| **-l** | indicate sequential pivot language translation level.<ul><li>To have **1-pivot** language `-l 1`. e.g. English-Russian-English, English-Italian-Emglish.</li><li>To have **2-pivot** language `-l 2`. e.g. English-Russian-French-English, English-Japanese-Spanish-Emglish.</li><li> By default `-l 0` apply both 1-pivot and 2-pivot language translation.</li></ul>|
| **-c** | cut-off integer that indicate how many parpahrases to select, e.g. `-c 3` will only select top highest 3 semantically related parpahrases and drop the rest.|


This will save the generated paraphrases in the **result** folder. The **result** file display the paraphrases in Python dictionary(**Key** -- initial sentences and **Value** -- list of paraphrases) following the given order:
>- Paraphrases generated by Translation API,
>- Paraphrases after filtration with [Universal Sentence Encoding](https://tfhub.dev/google/universal-sentence-encoder/4)
>- Paraphrases after filtration with [BERT](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)
>- Paraphrases after duplication with [BERT](https://huggingface.co/transformers/model_doc/bert.html#bertmodel)




## References
<a id="1">[1]</a> Weir, Nathaniel and Crotty, Andrew and Galakatos, Alex and others. "DBPal: Weak Supervision for Learning a Natural Language Interface to Databases." _arXiv preprint arXiv:1909.06182_ (2019).