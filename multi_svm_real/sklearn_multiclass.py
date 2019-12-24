# -*- coding: utf-8 -*-
# @Time     :2019/11/19 18:01
# @Author   :XiaoMa
# @File     :sklearn_multiclass.py.py

from sklearn import multiclass, svm
from sklearn.svm import SVC
import pickle

def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    """
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    """
    y_pred_train = None
    y_pred_test = None
    # using random_state=12345 for reproductivity
    # svm_model = svm.LinearSVC(random_state=12345)
    svm_model=SVC(verbose=1)
    # print(X_train)
    if mode == 'ovr':
        ovr_model = multiclass.OneVsRestClassifier(svm_model)
        ovr_model.fit(X_train, y_train)
        # print(ovr_model)
        y_pred_train = ovr_model.predict(X_train)
        # y_pred_test = ovr_model.predict(X_test)
        pickle.dump(ovr_model,open('ovr_model.pkl','wb'))
    return y_pred_train #, y_pred_test

