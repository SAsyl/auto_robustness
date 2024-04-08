import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string

from pymorphy2 import MorphAnalyzer

import pandas as pd
import re
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity
# from gensim import models

from catboost import CatBoostClassifier, Pool

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, \
    SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from collections import defaultdict

from translate import Translator

import textattack
from textattack.augmentation import WordNetAugmenter, EmbeddingAugmenter, EasyDataAugmenter, \
    CharSwapAugmenter, DeletionAugmenter
from textattack.models.wrappers import ModelWrapper, SklearnModelWrapper

import pickle

from configparser import ConfigParser

import random


config_file = sys.argv[1]
# print(config_file)
config = ConfigParser()
# config.read(f"./../configs/labeled_tweets_ru_CNN.ini")
config.read(f"./../configs/{config_file}")

selected_model_type = config["main parameters"]["model_type"]
selected_attack_type = config["main parameters"]["attack_type"]
dataset_filename = config["main parameters"]["dataset_filename"]
selected_language = config["main parameters"]["language"]
selected_vectorization_type = config["main parameters"]["vectorization_type"]
selected_augm_method = config["adversarial method options"]["augmenter"]

# print(selected_model_type, selected_attack_type, dataset_filename, selected_language, selected_vectorization_type)

num_words = 1000
max_review_len = 20

if selected_language == "ru":
    stop_words_dir = "./../stopwords_ru.txt"
    language = "russian"
elif selected_language == "eng":
    stop_words_dir = "./../stopwords_eng.txt"
    language = "english"

with open(stop_words_dir, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    stop_words = []
    for line in lines:
        stop_words.append(line.strip())
    # print(stop_words)

# morph = MorphAnalyzer()
snowball = SnowballStemmer(language=language)
translator_ru_2_eng = Translator(to_lang='en', from_lang='ru')
translator_eng_2_ru = Translator(to_lang='ru', from_lang='en')


def nomalize_and_remove_stopwords(line):
    tokenized = word_tokenize(line, language=language)

    tokenized = [i for i in tokenized if i not in string.punctuation]
    tokenized = [i for i in tokenized if i not in stop_words]
    tokenized = [snowball.stem(i) for i in tokenized]

    # filtered = " ".join([morph.normal_forms(w.strip())[0] for w in tokenized if not w.lower() in stop_words])
    # filtered = re.sub(r'[^\w\s]', '', filtered)
    # if selected_language == "ru":
    #     filtered = re.sub(r'[^а-яА-Я]', ' ', filtered)
    # elif selected_language == "eng":
    #     filtered = re.sub(r'[^a-zA-Z]', ' ', filtered)
    filtered = ' '.join([token for token in tokenized if len(token) > 2])

    return filtered


def most_frequently_words(dataset, top=10, bottom=10):
    word_freq = defaultdict(int)
    for tokens in dataset:
        for token in tokens:
            word_freq[token] += 1

    print(f"Unique words: {len(word_freq)}")

    sorted_word_freq = sorted(word_freq, key=word_freq.get, reverse=True)

    topn_frequented = sorted_word_freq[:top]
    lastn_frequented = sorted_word_freq[-bottom:]

    print(f"Most popular words: {topn_frequented}\nMost unpopular words: {lastn_frequented}")

    # return topn_frequented, lastn_frequented


def preparation_ds(filename):
    global imported_dataset
    imported_dataset = pd.read_csv(f'./../datasets/{filename}',
                                   # header=None,
                                   # names=['Class', 'Review'],
                                   # index_col=0
                                   )
    # if "translated_text" not in imported_dataset.columns and selected_language == "ru":
    #     translated_texts = []
    #     for text in df['text']:
    #         translated_text = translator_ru_2_eng.translate(text)
    #         translated_texts.append(translated_text)
    #
    #     # Create a new DataFrame with translated texts
    #     translated_df = pd.DataFrame({'text': translated_texts, 'toxic': imported_dataset['toxic']})
    #
    #     # Save the translated dataset to a new file
    #     # translated_df.to_csv('english_dataset.csv', index=False)

    if "normalized" not in filename:
        imported_dataset.rename(columns={f'{config["columns name"]["text_column"]}': 'text'}, inplace=True)
        imported_dataset.rename(columns={f'{config["columns name"]["label_column"]}': 'toxic'}, inplace=True)

        imported_dataset = imported_dataset.dropna()
        imported_dataset = imported_dataset.drop_duplicates()

        imported_dataset['text'] = imported_dataset['text'].apply(lambda x: nomalize_and_remove_stopwords(str(x)))
        print(f'''
            {'—' * 20}
            Dataset normalized
            {'—' * 20}''')

        imported_dataset['splitted'] = imported_dataset.text.apply(lambda x: str(x).split(' '))

        new_dataset_filename = f"{dataset_filename[:-4]}_normalized.csv"
        imported_dataset.to_csv(f'./../datasets/{new_dataset_filename}')

        config["main parameters"]["dataset_filename"] = new_dataset_filename
        config["columns name"]["text_column"] = "text"
        config["columns name"]["label_column"] = "toxic"
        with open(f"./../configs/{config_file}", "w") as cfg_file:
            config.write(cfg_file)

    print(f"The shape of dataset: {imported_dataset.shape[0]} x {imported_dataset.shape[1]}")

    return imported_dataset


def print_metrics(train_true, train_pred, test_true, test_pred):
    print("————————————————————————")
    print("Training model:")
    print("————————————————————————")
    print(f"Precision (Train): {round(precision_score(train_true, train_pred) * 100, 2)}%")
    print(f"Recall (Train): {round(recall_score(train_true, train_pred) * 100, 2)}%")
    print(f"Accuracy (Train): {round(accuracy_score(train_true, train_pred) * 100, 2)}%")
    print(f"F1 (Train): {round(f1_score(train_true, train_pred) * 100, 2)}%")
    print(f"AUC_ROC (Train): {round(roc_auc_score(train_true, train_pred) * 100, 2)}%")

    print("\n————————————————————————")
    print("Test model:")
    print("————————————————————————")
    print(f"Precision (Test): {round(precision_score(test_true, test_pred) * 100, 2)}%")
    print(f"Recall (Test): {round(recall_score(test_true, test_pred) * 100, 2)}%")
    print(f"Accuracy (Test): {round(accuracy_score(test_true, test_pred) * 100, 2)}%")
    print(f"F1 (Test): {round(f1_score(test_true, test_pred) * 100, 2)}%")
    print(f"AUC_ROC (Test): {round(roc_auc_score(test_true, test_pred) * 100, 2)}%")


def create_model(x_train, y_train,
                 x_test, y_test,
                 model_type,
                 optimizer='adam',
                 loss_function='binary_crossentropy',
                 vectorization_type="WordBug"):
    global model, history, x_train_tokenized_vectorized, x_test_tokenized_vectorized

    if model_type == "CB":
        model_save_path = f'./pipeline/models/Model_CB'
    elif model_type == "LR":
        model_save_path = f'./pipeline/models/Model_LR.sav'
    else:
        model_save_path = f'./pipeline/models/{model_type}/{dataset_filename[:-4]}'

    model_structure_path = f'./pipeline/models/structures/{model_type}_structure.png'
    if vectorization_type == "WordBug":
        wb_tokenizer = Tokenizer(num_words=num_words)
        wb_tokenizer.fit_on_texts(x_train)

        sequences = wb_tokenizer.texts_to_sequences(x_train)
        x_train_tokenized_vectorized = pad_sequences(sequences, maxlen=max_review_len)
    elif vectorization_type == "TFIDF1" or vectorization_type == "TFIDF":
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        x_tf_idf = tf_idf_vectorizer.fit_transform(x_train).toarray()
        x_train_tokenized_vectorized = pad_sequences(x_tf_idf, maxlen=max_review_len, dtype='float32')
    elif vectorization_type == "TFIDF2":
        tf_idf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
        x_tf_idf = tf_idf_vectorizer.fit_transform(x_train).toarray()
        x_train_tokenized_vectorized = pad_sequences(x_tf_idf, maxlen=max_review_len, dtype='float32')

    if model_type == "CNN":
        model = Sequential()
        model.add(Embedding(num_words, 64, input_length=max_review_len))
        model.add(Conv1D(250, 5, padding='valid', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "RNN":
        model = Sequential()
        model.add(Embedding(num_words, 2, input_length=max_review_len))
        model.add(SimpleRNN(8))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "LSTM":
        model = Sequential()
        model.add(Embedding(num_words, 8, input_length=max_review_len))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "GRU":
        model = Sequential()
        model.add(Embedding(num_words, 8, input_length=max_review_len))
        model.add(GRU(32))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "LR":
        model = LogisticRegression(solver=config["model creation options"]["solver"],
                                   max_iter=10000,
                                   n_jobs=-1,
                                   random_state=12345)
        model.fit(x_train_tokenized_vectorized, y_train)

        pickle.dump(model, open(model_save_path, 'wb'))
    elif model_type == "CB":
        model = CatBoostClassifier(iterations=int(config["model train options"]["iterations"]),
                                   learning_rate=float(config["model train options"]["learning_rate"]),
                                   depth=int(config["model train options"]["depth"]))
        model.fit(x_train_tokenized_vectorized, y_train)

    if model_type in ["CNN", "RNN", "LSTM", "GRU"]:
        # plot_model(model, to_file=model_structure_path, show_shapes=True, show_layer_names=True)

        model.compile(optimizer=config["model creation options"]["optimizer"],
                      loss=loss_function,
                      metrics=['accuracy'])

        checkpoint_callback = ModelCheckpoint(model_save_path,
                                              monitor='val_accuracy',
                                              save_best_only=True,
                                              verbose=1)
        history = model.fit(x_train_tokenized_vectorized,
                            y_train,
                            epochs=int(config["model train options"]["epochs"]),
                            batch_size=int(config["model train options"]["batch_size"]),
                            validation_split=0.1,
                            callbacks=[checkpoint_callback])
        model.save(model_save_path)

        plt.plot(history.history['accuracy'],
                 label='Доля верных ответов на обучающем наборе')
        plt.plot(history.history['val_accuracy'],
                 label='Доля верных ответов на проверочном наборе')
        plt.xlabel('Эпоха обучения')
        plt.ylabel('Доля верных ответов')
        plt.legend()
        # plt.show()

        plt.savefig(f'./pipeline/metrics/{model_type}/{dataset_filename[:-4]}.png')
        plt.close()
    elif model_type == "LR":
        print(f"Model is LogisticRegression")
    elif model_type == "CB":
        print(f"Model is CatBoost")

    return model_save_path, model


def model_score(x_train, y_train, x_test, y_test, loaded_model, model_type, vectorization_type):
    if vectorization_type == "WordBug":
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(x_train)

        sequences = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(sequences, maxlen=max_review_len)

        test_sequences = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(test_sequences, maxlen=max_review_len)
    elif vectorization_type == "TFIDF1" or vectorization_type == "TFIDF":
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        x_train_td_idf = vectorizer.fit_transform(x_train).toarray()
        x_test_td_idf = vectorizer.transform(x_test).toarray()

        x_train = pad_sequences(x_train_td_idf, maxlen=max_review_len, dtype='float32')
        x_test = pad_sequences(x_test_td_idf, maxlen=max_review_len, dtype='float32')
    elif vectorization_type == "TFIDF2":
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
        x_train_td_idf = vectorizer.fit_transform(x_train).toarray()
        x_test_td_idf = vectorizer.transform(x_test).toarray()

        x_train = pad_sequences(x_train_td_idf, maxlen=max_review_len, dtype='float32')
        x_test = pad_sequences(x_test_td_idf, maxlen=max_review_len, dtype='float32')

    if model_type in ["CNN", "RNN", "LSTM", "GRU"]:
        model_scores = loaded_model.evaluate(x_test, y_test, verbose=1)
        print(f"Accuracy on test: {model_scores[1]}")

    elif model_type == "LR":
        # predict_logreg_base_proba = loaded_model.predict_proba(x_test)
        # print(f"Model predict proba: {predict_logreg_base_proba}")

        model_train_pred = loaded_model.predict(x_train)
        model_test_pred = loaded_model.predict(x_test)
        # print(f"Accuracy on train: {loaded_model.score(x_train, y_train)}")
        print(f"Accuracy on test: {loaded_model.score(x_test, y_test)}")

        print_metrics(y_train, model_train_pred, y_test, model_test_pred)
    elif model_type == "CB":
        y_pred_train = loaded_model.predict(x_train)
        y_pred_test = loaded_model.predict(x_test)

        # print(sum(y_pred), len(y_pred))
        y_train_listed = y_train.values.tolist()
        y_test_listed = y_test.values.tolist()
        # print(sum(y_test_listed), len(y_test_listed))
        acc_score_train = accuracy_score(y_pred_train, y_train_listed)
        acc_score_test = accuracy_score(y_pred_test, y_test_listed)
        # print(f"Accuracy on train: {acc_score_train}")
        print(f"Accuracy on test: {acc_score_test}")


def train_test_split_dataset(texts, labels, test_size=0.3, splitted=None):
    # if selected_vectorization_type == "WordBug":
    #     x_train, x_test, y_train, y_test = train_test_split(texts, labels,
    #                                                         test_size=float(
    #                                                             config["dataset splitting parameters"]["test_size"]),
    #                                                         random_state=42,
    #                                                         shuffle=True)
    # elif selected_vectorization_type == "TFIDF":
    #     x_train, x_test, y_train, y_test = train_test_split(splitted, labels,
    #                                                         test_size=float(
    #                                                             config["dataset splitting parameters"]["test_size"]),
    #                                                         random_state=42,
    #                                                         shuffle=True)

    x_train, x_test, y_train, y_test = train_test_split(texts, labels,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


def apply_augmentation_method(texts):
    texts_augmented_examples = []
    dataset_size = texts.shape[0]

    i = 0
    if selected_augm_method == "Deletion":
        augm_method = DeletionAugmenter()
        for text in texts:
            i += 1
            augmented_text = augm_method.augment(str(text))
            texts_augmented_examples.append(augmented_text)
            if i % 50 == 0:
                print(f"Augmented {int(i / dataset_size * 100)}%!")
    elif selected_augm_method == "WordNet":
        augm_method = WordNetAugmenter()
        for text in texts:
            i += 1
            augmented_text = augm_method.augment(str(text))
            texts_augmented_examples.append(augmented_text)
            if i % 50 == 0:
                print(f"Augmented {int(i / dataset_size * 100)}%!")
    elif selected_augm_method == "Embedding":
        augm_method = EmbeddingAugmenter()
        for text in texts:
            i += 1
            augmented_text = augm_method.augment(str(text))
            texts_augmented_examples.append(augmented_text)
            if i % 50 == 0:
                print(f"Augmented {int(i / dataset_size * 100)}%!")
    elif selected_augm_method == "EasyData":
        augm_method = EasyDataAugmenter()
        for text in texts:
            augmented_texts = augm_method.augment(str(text))
            random_index = random.randint(0, len(augmented_texts) - 1)
            augmented_text = augmented_texts[random_index]
            texts_augmented_examples.append(augmented_text)
            if i % 50 == 0:
                print(f"Augmented {int(i / dataset_size * 100)}%!")
    elif selected_augm_method == "CharSwap":
        augm_method = CharSwapAugmenter()
        for text in texts:
            i += 1
            augmented_text = augm_method.augment(str(text))
            texts_augmented_examples.append(augmented_text)
            if i % 50 == 0:
                print(f"Augmented {int(i / dataset_size * 100)}%!")

    return texts_augmented_examples


# Evaluating model
# model.load_weights(model_save_path)

# loaded_model = pickle.load(open(model_LR, 'rb'))
# result_test = loaded_model.score(tf_idf_test, y_test)
# print(result_test)

# model = CatBoostClassifier()
# model.load_model(model_CB)

# baseline_value = y_train.mean()
# print(baseline_value)

# test_sequences = tokenizer.texts_to_sequences(x_test)
# x_test = pad_sequences(test_sequences, maxlen=max_review_len)
# print(model.evaluate(x_test, y_test, verbose=1))
#
# # Evaluating on out own comment
# text = ['''Данный продукт был приобретен мной 2 недели назад. Спустя это время могу сказать,
# что он неплох, удобен в использовании, очень гибок и эластисен. Однако бывали моменты, когда он
# выскальзывал из рук.
# ''', "Мне понравилась его идея", "Это было хуево, как мне стыдно!"]
#
# text_sequence = tokenizer.texts_to_sequences(text)
# print(text_sequence)
#
# text_pad_seq = pad_sequences(text_sequence, maxlen=max_review_len)
# print(text_pad_seq)
#
# result = model.predict(text_pad_seq)
# print(result)
#
# for res in result:
#     print("Positive" if res > 0.5 else "Negative")

df = preparation_ds(dataset_filename)

# most_frequently_words(df['splitted'], 10, 10)


x_train, x_adv_test, y_train, y_adv_test = train_test_split_dataset(df.text, df.toxic, float(config["dataset splitting parameters"]["test_size"]))

x_adv, x_test, y_adv, y_test = train_test_split_dataset(x_adv_test, y_adv_test, float(config["adversarial method options"]["test_size"]))


print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),
                                                                             (len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
                                                                            (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))
print("Adversarial set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_adv),
                                                                             (len(x_adv[y_adv == 0]) / (len(x_adv)*1.))*100,
                                                                            (len(x_adv[y_adv == 1]) / (len(x_adv)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),
                                                                             (len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
                                                                            (len(x_test[y_test == 1]) / (len(x_test)*1.))*100))

# print(x_train, y_train, sep='\n')
# print(x_test, y_test, sep='\n')
# print(sum(y_train), len(y_train)-sum(y_train))
# print(sum(y_test), len(y_test)-sum(y_test))

# save_path, myModel = create_model(x_train, y_train,
#                                   x_adv_test, y_adv_test,
#                                   model_type=selected_model_type,
#                                   vectorization_type=selected_vectorization_type)

model_load_path = f'./pipeline/models/{selected_model_type}/{dataset_filename[:-4]}'
myModel = load_model(model_load_path)

model_score(x_train, y_train, x_adv_test, y_adv_test, myModel, selected_model_type, selected_vectorization_type)

# x_augmented = apply_augmentation_method(x_adv)
# print(x_augmented)

# model_score(x_train, y_train, x_augmented, y_adv, myModel, selected_model_type, selected_vectorization_type)

# if selected_vectorization_type == "WordBug":
#     vectorizer = Tokenizer(num_words=num_words)
#     vectorizer.fit_on_texts(x_train)
#
#     sequences = vectorizer.texts_to_sequences(x_train)
#     x_train = pad_sequences(sequences, maxlen=max_review_len)
#
#     test_sequences = vectorizer.texts_to_sequences(x_test)
#     x_test = pad_sequences(test_sequences, maxlen=max_review_len)
# elif selected_vectorization_type == "TFIDF1" or vectorization_type == "TFIDF:
#     vectorizer = TfidfVectorizer(ngram_range=(1, 1))
#     x_train_td_idf = vectorizer.fit_transform(x_train).toarray()
#     x_test_td_idf = vectorizer.transform(x_test).toarray()
#
#     x_train = pad_sequences(x_train_td_idf, maxlen=max_review_len, dtype='float32')
#     x_test = pad_sequences(x_test_td_idf, maxlen=max_review_len, dtype='float32')
# elif selected_vectorization_type == "TFIDF2":
#     vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
#     x_train_td_idf = vectorizer.fit_transform(x_train).toarray()
#     x_test_td_idf = vectorizer.transform(x_test).toarray()
#
#     x_train = pad_sequences(x_train_td_idf, maxlen=max_review_len, dtype='float32')
#     x_test = pad_sequences(x_test_td_idf, maxlen=max_review_len, dtype='float32')
