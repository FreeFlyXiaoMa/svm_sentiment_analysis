# -*- coding: utf-8 -*-
# @Time     :2019/11/19 18:14
# @Author   :XiaoMa
# @File     :svm_multiclass_train.py

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics

from sklearn_multiclass import sklearn_multiclass_prediction
from preprocession import load_file_and_split

from gensim.models.word2vec import Word2Vec

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


def get_train_vecs(x_all_sentences, x_train_sentences, x_test_sentences):
    print('开始训练词向量！！！')
    # 将每个词用300个维度向量化
    n_dim = 300
    # 初始化word2vec模型，默认为cbow模型
    comment_w2v = Word2Vec(size=n_dim, min_count=1)
    # 确定word2vec的词表
    comment_w2v.build_vocab(x_all_sentences)
    # 训练word2vec并模型
    comment_w2v.train(x_all_sentences, total_examples=comment_w2v.corpus_count, epochs=100)
    # 保存模型
    comment_w2v.save('w2v_model.pkl')
    # 训练数据的向量化
    train_vectors = np.concatenate([build_word_vector(z, n_dim, comment_w2v) for z in x_train_sentences])
    # 测试数据的向量化
    test_vectors = np.concatenate([build_word_vector(z, n_dim, comment_w2v) for z in x_test_sentences])
    # np.save('test_vectors.npy', test_vectors)
    return train_vectors, test_vectors


def main():

    print('Training Sklearn OVR...')
    x, x_train, x_test, y_train, y_test = load_file_and_split()
    train_vecs, test_vecs = get_train_vecs(x, x_train, x_test)
    # 均值方差归一化向量
    # standardScaler = StandardScaler()
    # standardScaler.fit(train_vecs)
    # train_vecs = standardScaler.transform(train_vecs)
    # print('---',len(train_vecs))
    #
    # test_vecs = standardScaler.transform(test_vecs)

    y_pred_train = sklearn_multiclass_prediction(
        'ovr', train_vecs, y_train, test_vecs)
    print('SVM Accuracy (train):',
          metrics.accuracy_score(y_train, y_pred_train))


if __name__=='__main__':
    main()

