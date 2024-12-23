from src.sentimentanalysis_tf.logger import logger
from src.sentimentanalysis_tf.classifier import ml_classifier, keras_lstm
from src.sentimentanalysis_tf.utils import stem_text, save_object, load_object

from keras._tf_keras.keras.saving import load_model

from flask import Flask, render_template, request

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import os

logger=logger()

nltk.download('stopwords')


stopwords = set(stopwords.words('english'))
encoder = load_object(os.path.join("objects","encoder.pkl"))
vectorizer = load_object(os.path.join("objects","vectorizer.pkl"))
trained_model = load_model(os.path.join("objects","trained_model.keras"))

logger.info("<<<<< Below have been loaded successfully: >>>>>\n- stopwords\n- encoder\n- vectorizer\n- pre_trained_model")


app =Flask(__name__)

input_size=11000
output_size=150
input_length=300

@app.route("/predict_sentiment/", methods=["GET","POST"])
def analyze_emotion():
    if request.method == 'POST':
        comment = request.form.get('comment')
        prediction, probability = keras_lstm.predict_emotion(model=trained_model,sentence=comment, stemmer=PorterStemmer(), stopwords=stopwords, vocab_size=input_size, max_len=input_length, encoder=encoder)
        #decoded_pred = encoder.inverse_transform(prediction)
        logger.info(f"<<<<< Sentence: {comment} >>>>>\nOutput:\nSentiment:{type(prediction)}, with probability {probability}\n{'=='*50}")
        return render_template("index.html", sentiment=prediction)

    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)

