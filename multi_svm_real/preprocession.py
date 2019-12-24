import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

# 分词和去掉停用词
def processing_word(x, stop_words):
    word_list=['对会会','行行行','方便','网银','你好','起来','静音','请问','开车','喂喂','误导','上次','人太长',
               '没人能','算开','杨对','空对','现对','不多人','没人收','难要','老不来','车不开','长不长','好快','取不来'
               ,'开不来']

    for word in word_list:
        jieba.suggest_freq(word,True)   #--调整词语的出现频率

    cut_word = jieba.cut(x.strip())
    # word = [word for word in cut_word if word not in stop_words]
    word=[word for word in cut_word]
    # print('停用词处理结果：',word)
    return word


def get_stop_words():

    # stop_words_list = []
    # with open('stop_words.txt', 'r',encoding='GBK') as stop_words_file:
    #     for line in stop_words_file:
    #         stop_words_list.append(line.strip())
    # return stop_words_list
    return None

def load_file_and_split():

    # positive_comment = pd.read_excel('comment.xls', header=None, heet_name='positive_comment')
    # negative_comment = pd.read_excel('comment.xls', header=None, sheet_name='negative_comment')
    stop_words = get_stop_words()
    # positive_comment['words'] = positive_comment[0].apply(processing_word, args=(stop_words,))
    # negative_comment['words'] = negative_comment[0].apply(processing_word, args=(stop_words,))
    # x = np.concatenate((positive_comment['words'], negative_comment['words']))
    # y = np.concatenate((np.ones(len(positive_comment)), np.zeros(len(negative_comment))))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    train=pd.read_csv('train.csv')
    val=pd.read_csv('val.csv')


    train['用户']=train['用户'].apply(processing_word,args=(stop_words,))
    val['用户']=val['用户'].apply(processing_word,args=(stop_words,))

    corpus=pd.read_csv('corpus.csv')
    # x=np.concatenate((train['用户'],val['用户']))
    x=corpus['用户']
    x=x.apply(processing_word,args=(stop_words,))
    print('语料库的大小',len(x))
    x_train=train['用户']
    y_train=train['polar'].astype(int)

    x_test=val['用户']
    y_test=val['polar'].astype(int)
    # print(train.head(10))

    return x, x_train, x_test, y_train, y_test

