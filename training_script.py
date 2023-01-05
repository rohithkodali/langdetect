"""
Author : Rohith Gowtham
Date : 05/05/2021 16:43:19 PM
To build a language detection classifier using BPE and logistic regression
Preprocessing steps:
Combine all the data into a single file and train the tokenizer on it. Remove the Punctuation and other special characters except single quotes.
Build the BPE (Byte Pair Encoding) or WPE (Word Pair Encoding) tokenizer using the Huggingface Tokenizers library.
Apply BPE/WPE to the input text to generate a sequence of BPE tokens.

Extract features from the BPE tokens:
Create a vocabulary of all the BPE tokens in the training data one line as a training example.
For each training example, create a feature vector that represents the presence or absence of each BPE token in the vocabulary.
Data should be in the format of a list of tuples (features, label) where features is a dictionary of BPE tokens and label is the language of the text.

Train a logistic regression classifier:
Use the feature vectors and labels (i.e., the language of each training example) to train a logistic regression classifier. we can even replace this with SVM or any other classifier.

Evaluate the classifier:
Use the trained classifier to predict the language of new input examples and evaluate its performance using metrics such as accuracy, precision, and recall.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import glob, os
import re
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"
unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens


def prepare_tokenizer_trainer(algorithm="bpe", vocab_size=10000):
    '''
    Prepares the tokenizer and trainer for training.
    :param algorithm: Choose between bpe and wordpiece
    :param vocab_size: Choose the vocabulary size for the tokenizer algorithm
    :return: tokenizer and trainer
    '''
    if algorithm == "bpe":
        print("Training BPE tokenizer")
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    elif algorithm == "wpe":
        print("Using WordPiece")
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer


def train_tokenizer(files):
    """
    Trains the tokenizer on the files provided.
    :param files: need the file with all langauge data to train the tokenizer
    :return: Tokenizer will be saved to disk and can be loaded and the object is returned
    """
    tokenizer, trainer = prepare_tokenizer_trainer(algorithm="bpe", vocab_size=10000)
    tokenizer.train(files, trainer)  # training the tokenzier
    tokenizer.save("models/tokenizer-trained.json")  # save the tokenizer to disk
    tokenizer = Tokenizer.from_file("models/tokenizer-trained.json")  # laod the tokenizer from disk only for demonstration purpose
    return tokenizer

def tokenize(input_string, tokenizer):
    '''
    Tokenizes the input string using the tokenizer provided.
    :param input_string: text to be tokenized
    :param tokenizer: tokenizer instance to be used for tokenization
    :return: tokenized string
    '''
    output = tokenizer.encode(input_string)
    return output

#please add any new langauge in the folder with files in it and add the path to the list below

languages = ['english', 'french', 'german']  # list of languages in the repo as of now
outfile=open("alldata","w") # file to write all the data to
train_data = []
for language in languages:
    for filepath in glob.glob(os.path.join(language, '*')):
        data=open(filepath, 'r', encoding='utf-8').read()
        for line in data.split('\n'):
            if line:
                line = re.sub(r'[^\w\s\']','',line).lower()  #normalized data by removing special characters and converting to lower case
                outfile.write(line+"\n")
outfile.close()

#End of data preparation for tokenizer training

print("Training tokenizer...")
trained_tokenizer = train_tokenizer(["alldata"])
print('tokenizer trained and loaded')

#preparing data for training classifier format (features, label)
for language in languages:
    for filepath in glob.glob(os.path.join(language, '*')):
        data = open(filepath, 'r').read()
        for line in data.split('\n'):
            # check if line is not empty
            if line:
                # normalize the data by removing special characters and converting to lower case
                line = re.sub(r'[^\w\s\']', '', line).lower()
                output = trained_tokenizer.encode(line)
                print(output.tokens, language)
                train_data.append((output.tokens, language)) # adding the tokenized data and the language to the list

# preparing the data for training the classifier
features = []
labels = []
for item in train_data:
    features.append(' '.join(item[0]))
    labels.append(item[1])
# Vectorize the features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(features)

#Generating the unique labels for the classifier
labels, uniq = np.unique(labels, return_inverse=True)

# Fit a logistic regression model
clf = LogisticRegression(random_state=0, multi_class='auto').fit(X, uniq)
# save the model

pickle.dump(clf, open('models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
pickle.dump(trained_tokenizer, open('models/tokenizer.pkl', 'wb'))
# save labels to pickle
pickle.dump(labels, open('labels.pkl', 'wb'))

# Test the model on some new data  a sample test case to see the output
test = "Un des quarante ou cinquante témoins entendus par la commission d'enquête a dit"
test = re.sub(r'[^\w\s\']', '', test).lower()
test_data = tokenize(test, trained_tokenizer)
print(test_data.tokens)
test_features = vectorizer.transform([' '.join(test_data.tokens)])
prediction = clf.predict(test_features)
print(labels[prediction])  # Outputs the predicted language (e.g., 'english')
