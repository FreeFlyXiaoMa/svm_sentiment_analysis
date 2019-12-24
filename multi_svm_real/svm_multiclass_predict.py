# -*- coding: utf-8 -*-
# @Time     :2019/11/20 10:45
# @Author   :XiaoMa
# @File     :svm_multiclass_predict.py

import numpy  as np
import pickle
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import StandardScaler
import jieba
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

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
    # print('vec::',vec)
    return vec

# 分词和去掉停用词
def processing_word(x, stop_words):
    word_list=['对会会','行行行','方便','网银','你好','起来','静音','请问','开车','喂喂','误导','上次','人太长',
               '没人能','算开','杨对','空对','现对','不多人','没人收','难要','老不来','车不开','长不长'
               ,'开不来']

    for word in word_list:
        jieba.suggest_freq(word,True)   #--调整词语的出现频率

    cut_word = jieba.cut(x.strip())
    # word = [word for word in cut_word if word not in stop_words]
    word=[word for word in cut_word]
    # print('停用词处理结果：',word)
    return word

def get_test_vectors(x_test_sentences):
    # print('开始加载词向量！！')
    n_dim=300
    w2v_model=Word2Vec.load('w2v_model.pkl')
    # w2v_model.most_similar(positive='单词')
    x_test_sentences=[jieba.lcut(x_test_sentences)]
    # print('x_test_sentences:',x_test_sentences)
    test_vectors=np.concatenate([build_word_vector(z,n_dim,w2v_model) for z in x_test_sentences])

    return test_vectors

def predict(sent):
    test_vectors = get_test_vectors(sent)
    # print(test_vectors)
    # 均值方差归一化
    # standardScaler = StandardScaler()
    # standardScaler.fit(test_vectors)
    # test_vectors = standardScaler.transform(test_vectors)
    # print('test_vectors:',test_vectors)
    ovr_model = pickle.load(open('ovr_model.pkl', 'rb'))
    y_pred=ovr_model.predict(test_vectors)

    return y_pred

if __name__ == '__main__':
    print('开始预测！！')
    import pandas as pd
    label_list=[]
    labe_pred_list=[]

    df_test=pd.read_csv('test.csv')
    for index in df_test.index:
        text=df_test['用户'][index]
        label=df_test['polar'][index]
        label_pred=predict(text)[0]
        label_list.append(label)
        labe_pred_list.append(label_pred)

    acc=round(accuracy_score(label_list,labe_pred_list),4)
    precision=round(precision_score(label_list,labe_pred_list,average='macro'),4)
    recall=round(recall_score(label_list,labe_pred_list,average='macro'),4)
    f1=round(f1_score(label_list,labe_pred_list,average='macro'),4)

    print('准确率：',acc,'精确率：',precision,'召回：',recall,'f1：',f1)