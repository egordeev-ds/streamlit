#imports
import re, string
import pickle

import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scorecardpy as scp
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

import streamlit as st

#del garbage
def preprocess(text):
    
    #нижний регистр
    text = text.lower()
    
    #удаление спецсимволов
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'[^\w\s]',' ',str(text).lower().strip())
    text = re.sub(r'\s+',' ',text)
    
    #удаление цифр
    text = re.sub(r"\d+", "", text, flags=re.UNICODE)
    
    #удаление лишних пробелов
    text = text.strip()
    
    return text

#del stopwords
def stopword(text):
    
    stopwords_ = stopwords.words('english')
    
    text_new = [word for word in word_tokenize(text) if word not in stopwords_]
    text_new = ' '.join(text_new)
    
    return text_new

###title
st.title("Build movies guesser")
                                 
### load button
uploaded_file = st.file_uploader("Choose a dataset")

### divider 1
st.divider()

if uploaded_file is not None:

    with st.spinner('Your model is preparing...'):
        
        #read
        df_init = pd.read_csv(uploaded_file, sep = ',')[:40]

        #proc_df
        def process_df(df_init):
            df_init['text'] = df_init['text'].apply(preprocess)
            df_init['text'] = df_init['text'].apply(stopword)
            return df_init
        df_init = process_df(df_init)

        #train_test_split
        def split_df(df_init):
            train, test = train_test_split(df_init, train_size=0.75, random_state = 42, stratify = df_init.label)
            x_train, y_train = train['text'], train['label']
            x_test, y_test = test['text'], test['label']
            return train, test, x_train, y_train, x_test, y_test
        train, test, x_train, y_train, x_test, y_test = split_df(df_init)

        #tf-idf
        def vectorize_df(x_train, x_test):
            vectorizer = TfidfVectorizer(use_idf=True, max_features = 200)
            x_train = pd.DataFrame.sparse.from_spmatrix(vectorizer.fit_transform(x_train), columns = vectorizer.get_feature_names_out())
            x_test  = pd.DataFrame.sparse.from_spmatrix(vectorizer.transform(x_test), columns = vectorizer.get_feature_names_out())
            x_train = pd.DataFrame(x_train.to_dict())
            x_test = pd.DataFrame(x_test.to_dict())
            return x_train, x_test
        x_train, x_test = vectorize_df(x_train, x_test)

        #model
        def build_model(x_train, y_train, x_test, y_test):
            model = CatBoostClassifier(
                random_state = 42,
                silent = True
            ).fit(x_train, y_train)
            train_true, train_pred = y_train.copy(), model.predict_proba(x_train)[:,1]
            test_true, test_pred = y_test.copy(), model.predict_proba(x_test)[:,1]
            train_perf = scp.perf_eva(train_true, train_pred, show_plot = False)['Gini']
            test_perf = scp.perf_eva(test_true, test_pred, show_plot = False)['Gini']
            return train_perf, test_perf, model
        train_perf, test_perf, model = build_model(x_train, y_train, x_test, y_test)


    ### result
    st.write('Congrats, you have trained your model. Below are the scores')
    st.write('Train Gini is:', round(train_perf*100,2), '%')
    st.write('Test Gini is:', round(test_perf*100,2), '%')

    ### divider2
    st.divider()

    ### download button
    st.download_button(
        "Download Model",
        data=pickle.dumps(model),
        file_name="model.pkl",
    )