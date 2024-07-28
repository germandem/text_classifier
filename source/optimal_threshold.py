import numpy as np


def expect_f1(y_prob, thres):

    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()

    f1 = 2 * tp / (2 * tp + fp + fn)

    return f1


def optimal_threshold(y_prob):

    y_prob = np.sort(y_prob)[::-1]
    fls = [expect_f1(y_prob, p) for p in y_prob]
    thres = y_prob[np.argmax(fls)]

    return thres, fls
