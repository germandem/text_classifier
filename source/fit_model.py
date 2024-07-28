import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from joblib import dump
import os

from source.text_processing import processed_text, get_vector_from_sentence
from source.optimal_threshold import optimal_threshold

from config import XGB_PARAMS, DATA_PATH


def fit_text_classifier_model():

    data_set = pd.read_csv(os.path.join(DATA_PATH, "dataset.csv"), index_col=0)
    data_set['normolized_text'] = data_set['text'].apply(lambda x: processed_text(x))

    vectors_df = get_vector_from_sentence(data_set["normolized_text"])

    X_train, X_test, y_train, y_test = train_test_split(
        vectors_df[vectors_df.columns[:300]], data_set.loc[vectors_df.index]['class'], test_size=0.2, random_state=42
    )

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    model = GridSearchCV(
        xgb_model,
        scoring="f1",
        param_grid=XGB_PARAMS,
        cv=kfold,
        verbose=4,
        n_jobs=6
    )

    model.fit(X_train, y_train)

    predict = model.best_estimator_.predict(X_test)
    predict_proba = model.best_estimator_.predict_proba(X_test)

    rate, _ = optimal_threshold(predict_proba[:, 1])

    print(rate)
    print(classification_report(y_test, list(map(lambda x: 1 if x > rate else 0, predict_proba[:, 1]))))
    print("accuracy: ", accuracy_score(y_test, list(map(lambda x: 1 if x > rate else 0, predict_proba[:, 1]))))
    print(confusion_matrix(y_test, list(map(lambda x: 1 if x > rate else 0, predict_proba[:, 1]))))

    # Save model
    dump(model, os.path.join(DATA_PATH, "model.joblib"))

    # Save metrics
    with open(os.path.join(DATA_PATH, 'metrics.txt'), 'w') as record_file:
        record_file.write(
            str(classification_report(y_test, list(map(lambda x: 1 if x > rate else 0, predict_proba[:, 1]))) + '\n')
        )
        record_file.write(
            str(confusion_matrix(y_test, list(map(lambda x: 1 if x > rate else 0, predict_proba[:, 1]))))
        )
    with open(os.path.join(DATA_PATH, 'optimal_rate.txt'), 'w') as optimal_rate:
        optimal_rate.write(str(rate))
