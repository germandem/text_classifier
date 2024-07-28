import os

PATH = os.path.dirname(__file__)

DATA_PATH = os.path.join(PATH, "data")

# https://github.com/natasha/navec
NAVEC_PATH = os.path.join(DATA_PATH, "navec_hudlit_v1_12B_500K_300d_100q.tar")

# https://github.com/natasha/corus/tree/master/data
NEWS_PATH = os.path.join(DATA_PATH, "lenta-ru-news.csv.gz")

XGB_PARAMS = {
    # "colsample_bytree": [0.7, 0.3, 0.1],
    # "gamma": [0, 0.5, 0.7],
    # "learning_rate": [0.1, 0.03, 0.003],  # default 0.1
    # "max_depth": [2, 4, 6],  # default 3
    # "n_estimators": [100, 300, 600, 900],  # default 100
    # "subsample": [0.6, 0.4, 0.2]
}

CLASSES = {
    "Спорт": 0,
    "Экономика": 1
}