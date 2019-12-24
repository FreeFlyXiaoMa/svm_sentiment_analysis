# -*- coding: utf-8 -*-
# @Time     :2019/11/14 11:47
# @Author   :XiaoMa
# @File     :2.py

import numpy as np
from xx.preprocession import processing_word, get_stop_words
# from xx.chat_analysis import build_word_vector
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

# 获得句子中所有词汇的向量，然后取平均值
def build_word_vector(text, size, comment_w2v):
    # print('走到这了')
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            # print('comment_w2v[word]:',word,comment_w2v[word])
            vec += comment_w2v[word].reshape((1, size))
            count += 1
        except KeyError:
            print(word ,'is not in vocabulary')
            continue
    if count != 0:
        vec /= count
    return vec

# 载入word2vec和svm训练好的模型做预测
def svm_predict(comment):
    n_dim = 300
    svm_clf = joblib.load('svm_model.pkl')
    w2v_model = Word2Vec.load('w2v_model.pkl')
    stop_words_list = get_stop_words()
    # print('stop_words_list:',stop_words_list)
    processed_comment = processing_word(comment, stop_words_list)
    comment_row = np.array(processed_comment).reshape(1, -1)
    comment_vectors = np.concatenate([build_word_vector(z, n_dim, w2v_model) for z in comment_row])
    # print('comment_vectors:',comment_vectors)
    predict_result = svm_clf.predict(comment_vectors)

    return predict_result

if __name__=='__main__':
    import pandas as pd
    df_test=pd.read_csv('test.csv')

    label_list=[]
    pred_list=[]
    for index in df_test.index:
        polar=df_test['polar'][index]
        text=df_test['用户'][index]
        if int(polar) !=0:
            predict_label = None
            label_list.append(polar)
            pred_result=svm_predict(text)
            if pred_result[0] <0.5:
#                 print(pred_result,',','negative',text)
                predict_label=-1
            elif 0.5 <= pred_result[0] < 1.0:
#                 print(pred_result,',','neural',text)
                predict_label=0
            elif pred_result[0] >= 1.0:
#                 print(pred_result,',','positive',text)
                predict_label=1
            pred_list.append(predict_label)
    acc=accuracy_score(label_list,pred_list)
    print('accuracy:',acc)
#     print(len(label_list),len(pred_list))

    #模型的准确率：0.79

