from wordcloud import WordCloud

import nltk
from nltk.stem import PorterStemmer

import re

import matplotlib.pyplot as plt
import matplotlib

from typing import Tuple
import pickle
import os
from src.sentimentanalysis_tf.logger import logger

logger = logger()


def show_words_cloud(wordcloud:WordCloud, emotion:str, figsize:Tuple[int,int]=(10,10), wc_size:int=15):
    plt.figure(figsize=figsize)
    plt.title(emotion + '_WordCloud', size=15)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def stem_text(text):
    """
    download stop words from nltk: nltk.download('stopwords')
    stemming text using nltk PorterStemmer
    """
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]"," ", text)
    text = text.lower()
    text = text.split()
    return " ".join([stemmer.stem(word) for word in text if word not in stopwords])

def save_object(file_name:str,object:object):
    dir_path = "objects"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    pickle.dump(object, open(os.path.join(dir_path,file_name), "wb"))
    logger.info(f"Object saved in {os.path.join(dir_path,file_name)}")

def load_object(file_name:str):
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)

if __name__=="__main__":
    __all__=["show_words_cloud","stem_text","save_object","load_object"]