from corus import load_lenta
import pandas as pd
import os

from config import NEWS_PATH, DATA_PATH, CLASSES


def download_news():

    records = load_lenta(NEWS_PATH)

    data_set = {"text": [], "class": []}
    
    for new in records:

        if new.topic in CLASSES.keys():
            data_set["text"] += [new.title]
            data_set["class"] += [new.topic]

    data_set = pd.DataFrame(data_set)

    data_set["class"] = data_set["class"].map(CLASSES)

    data_set = data_set.dropna(subset=["class"])

    print(data_set["class"].value_counts())

    data_set.to_csv(os.path.join(DATA_PATH, "dataset.csv"))
