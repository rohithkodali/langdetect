# langdetect
langauge detection algorithm that can be expandable to add any number of languages using BPE/WPE techinique.

For the sake of Large file issue the model needed by fasttext needed to be download separately

wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin ./

Run <b>pip install -r requirements.txt</b>  to install necessary modules

Brief document listing the key assumptions and design choices you've made (algorithm and models employed, languages supported, etc.)
* The approach used is generating the tokenizer using Byte pair encoding or word pair encoding technique from tokenizer.
* These will generate simple features of given text (sub word units) to build the model and for inference generation.
* we can use as many langauges as we can and this approach should support it, currently used english, french and german
* Adding a new langauge to it is simple, just create a folder with that language name and keep all text files in it, add the name of language in line 81 of training_cript.py over here languages = ['english', 'french', 'german'], it will handle the rest
* The important configuration we need to use is vocab_size parameter (for 3 langauges 256 size is enough the more languages the more vocab size is needed)
* Adding more data with a proper langauge tag we keep on improving performance of the model.


Simple architecture diagram.

                                  +------------+
                                  | Text dataset|
                                  +------------+
                                        |
                                        |
                                        v
                                  +------------+
                                  | BPE encoding|
                                  +------------+
                                        |
                                        |
                                        v
                                  +------------+
                                  | Word embeddings|
                                  +------------+
                                        |
                                        |
                                        v
                                  +------------+
                                  |Classifier model|
                                  +------------+
                                        |
                                        |
                                        v
                                  +------------+
                                  |Predicted language|
                                  +------------+


Simple API specification.
* The services can be started using python inference_server.py
* usage: curl -X 'POST' 'http://localhost:8888/langdetect/predict_framework?text=Il%20pr%C3%A9tend%20que%20sa%20pi%C3%A8ce%20n%27est%20pas%20morte' -H 'accept: application/json' -d ''

Instructions to test the system.
* Given above as a curl command or http://localhost:8888/langdetect/docs will open oepanapi document reference

* Time spent completing this assignment 
 Started Arounf 4:30 India Time and completed by the time i push this commit (~2 hour 15 min)
