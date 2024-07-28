import pandas as pd
from navec import Navec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2

from config import NAVEC_PATH

path_navec = NAVEC_PATH
navec = Navec.load(path_navec)

# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = stopwords.words("russian")


def get_words_list(text):

    return word_tokenize(text, language="russian")


def get_filtered_words_list(list_words):

    filtered_words = []

    for word in list_words:
        if word not in stop_words:
            filtered_words += [word]

    return filtered_words


def get_normolized_words_list(list_words):

    morph = pymorphy2.MorphAnalyzer()

    normal_words = []

    for word in list_words:
        normal_words += [morph.parse(word)[0].normal_form]

    return normal_words


def processed_text(text):

    text = (text.lower().replace("?", "").replace("  ", " "))

    text = get_normolized_words_list(get_filtered_words_list(get_words_list(text)))

    return text


def get_vector_from_sentence(series, group_func='max'):

    df = series.to_frame(name='normolized_text')

    vectors = {}

    for i, words in df['normolized_text'].items():

        vec = {}

        for num, word in enumerate(words):
            try:
                vec[num] = navec[word]
            except KeyError:
                continue

        if len(vec) == 0:
            continue

        vectors[i] = pd.DataFrame(vec).T.describe().loc[group_func].to_list()

    vectors_df = pd.DataFrame(vectors).T

    return vectors_df
