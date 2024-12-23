from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, classification_report

import keras as K
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.preprocessing.text import one_hot
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import models
import tensorflow as tf


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Any
import re
from tqdm import tqdm
from src.sentimentanalysis_tf.logger import logger 
from src.sentimentanalysis_tf.utils import stem_text

import os, sys


logger = logger()

@dataclass
class ml_classifier:

    X_train:Union[np.array, pd.Series]
    y_train:Union[np.array, pd.Series]

    X_test:Union[np.array, pd.Series]
    y_test:Union[np.array, pd.Series]

    def vectorize(self):
        tfidfvectorizer = TfidfVectorizer()
        X_train_tfidf = tfidfvectorizer.fit_transform(self.X_train)
        X_test_tfidf = tfidfvectorizer.transform(self.X_test)
        return X_train_tfidf, X_test_tfidf, tfidfvectorizer
    
    def train_and_predict(self)->dict:
        models={}
        classifier = {
            "MultinomialNB":MultinomialNB(),
            "LogisticRegression":LogisticRegression(),
            "RandomForestClassifier":RandomForestClassifier(),
            "Support Vector Machine":SVC()
        }

        X_train_tfidf, X_test_tfidf, _ = self.vectorize()

        for name, clf in classifier.items():
            logger.info(f"\n{'=='*50}{name}{'=='*50}")
            clf.fit(X_train_tfidf, self.y_train)
            y_pred = clf.predict(X_test_tfidf)
            acc_score = accuracy_score(self.y_test, y_pred)
            clf_report = classification_report(self.y_test, y_pred)
            logger.info(f"\n{'=='*50}{acc_score}{'=='*50}")
            logger.info("Classification Report")
            logger.info(clf_report)
            models.update({
                            f"{name}_model":clf,
                            f"{name}_accuracy_score":acc_score,
                            f"{name}_classification_score":clf_report
                            })
        return models

    def predict(self, init_model:Any):
        clf = init_model
        clf.fit(self.X_train, self.y_train)
        return clf.predict(self.X_train)
    
    @staticmethod
    def predict_emotion(init_model,input_text, vectorizor:TfidfVectorizer, encoder:LabelEncoder):
        model=init_model
        cleaned_text = stem_text(input_text)
        input_vectorizer = vectorizor.transform([cleaned_text])
        predicted_label = model.predict(input_vectorizer)[0]
        predicted_emotion = encoder.inverse_transform([predicted_label])[0]
        label=np.max(model.predict(input_vectorizer))

        return predicted_emotion, label

@dataclass
class keras_lstm:

    df:pd.DataFrame
    text_label:str
    cat_label:str

    def create_sequences(self, vocab_size:int, max_len:int, 
                         stemmer:Any, stopwords:Any)->pad_sequences:
        tqdm._instances.clear()
        corpus=[]
        for text in tqdm(self.df[self.text_label]):
            text = re.sub("[^a-zA-Z]"," ", text)
            text = text.lower()
            text = text.split()
            text = " ".join([stemmer.stem(word) for word in text if word not in stopwords])
            corpus.append(text)
        excoded_text = [one_hot(input_text=word, n=vocab_size) for word in corpus]
        self.sequences = pad_sequences(sequences=excoded_text, maxlen=max_len, padding="pre")
        return self.sequences
    
    def create_categories(self)->to_categorical:
        self.categories = to_categorical(self.df[self.cat_label])
        return self.categories
    
    def model(self, input_size:int, output_size:int, input_length:int, epochs:int, batch_size:int):
        model = Sequential()
        model.add(Embedding(input_dim=input_size, output_dim=output_size, input_length=input_length))
        model.add(Dropout(0.2))
        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(6, activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("<<<<< LSTM Model configured >>>>>")
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        logger.info("<<<<< Early stopping added to callbacks >>>>>")
        logger.info("<<<<< Training Initialized >>>>>")
        model.fit(self.sequences, self.categories, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
        logger.info("<<<<< Training Completed! >>>>>")
        return model
    
    @staticmethod
    def save_model(model:K.Model):
        dir_path = "objects"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        #model.save(filepath=os.path.join(dir_path,"model.h5"),overwrite=True)
        K.saving.save_model(model, filepath=os.path.join(dir_path,"model.keras"))

    @staticmethod
    def load_model(file_path:os.path):
        return K.saving.load_model(filepath=file_path)
    
    @staticmethod
    def predict_emotion(model:K.Model, sentence:str, stemmer:Any, stopwords:Any, vocab_size:int, max_len:int, encoder:LabelEncoder):
        tqdm._instances.clear()
        corpus=[]
        sentence = re.sub("[^a-zA-Z]"," ", sentence)
        sentence = sentence.lower()
        sentence = sentence.split()
        sentence = " ".join([stemmer.stem(word) for word in sentence if word not in stopwords])
        corpus.append(sentence)
        excoded_text = [one_hot(input_text=word, n=vocab_size) for word in corpus]
        sequences = pad_sequences(sequences=excoded_text, maxlen=max_len, padding="pre")
        #prediction = encoder.inverse_transform(np.argmax(model.predict(sequences), axis=1))[0]
        prediction = np.argmax(model.predict(sequences), axis=1)[0]
        probability = np.max(model.predict(sequences))*100
        print (f"The sentiment is {prediction}, with probability of {probability}")
        return prediction, probability
    
if __name__=="__main__":
    __all__=["ml_classifier","keras_lstm"]

