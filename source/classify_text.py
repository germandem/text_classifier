import pandas as pd
from joblib import load
from source.text_processing import get_vector_from_sentence, processed_text
import os

from config import DATA_PATH, CLASSES


def classify_text(text):

    class_to_name = {value: key for key, value in CLASSES.copy().items()}

    model = load(os.path.join(DATA_PATH, "model.joblib"))

    vectorised_text = get_vector_from_sentence(pd.Series([processed_text(text)]))

    predict = model.predict(vectorised_text)

    classification = class_to_name[predict[0]]

    return classification
