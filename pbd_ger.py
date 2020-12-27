import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
import argparse
from newspaper import Article
from tensorflow.keras.models import load_model
import pickle

import nltk 
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

mapping = ['Center', 'Left', 'Right']
MAX_SEQUENCE_LENGTH = 300

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        rec = tp/(K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
        return rec

    def precision(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        prec = tp/(K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
        return prec
    
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec + rec + K.epsilon()))

model = load_model('models/stacked_lstm.11-0.64.h5', custom_objects={'f1':f1})

with open('vocab.pkl', 'rb') as f:
    word_index = pickle.load(f)


def tokenize(texts):
    '''
    Split texts into tokens and get rid of punctuations and numbers
    '''
    if isinstance(texts, str):
        texts = [texts]
    texts_tokenized = [tokenizer.tokenize(t.lower()) for t in texts]
    return texts_tokenized


def vectorize(texts, word_index):
    texts = tokenize(texts)
    vect_texts = []
    for t in texts:
        vect_text = []
        for w in t:
            try:
                idx = word_index[w]
            except KeyError:
                idx = 0
            vect_text.append(idx)
        vect_texts.append(vect_text)
    vect_texts = np.array(vect_texts, dtype=object)

    #Padding 
    vect_texts = sequence.pad_sequences(vect_texts, maxlen=MAX_SEQUENCE_LENGTH)
    return vect_texts

def predict(url):
    a = Article(url)
    a.download()
    a.parse()
    
    vect_texts = vectorize(a.text, word_index)
    
    pred = model.predict(vectorize(a.text, word_index))[0]
    for i in range(len(pred)):
        print(mapping[i] + ': ' "{:.2%}".format(pred[i]))

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    p.add_argument('--url', action='store', dest='url', default=0, type=str, help='a link pointing to a news')
    args = p.parse_args()
    
    predict(args.url)