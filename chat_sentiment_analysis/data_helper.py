# -*- coding: utf-8 -*-
# @Time     :2019/11/11 13:24
# @Author   :XiaoMa
# @File     :data_helper.py
import pandas as pd

train=pd.read_csv('train.csv')
val=pd.read_csv('val.csv')

print(len(train))

for index in train.index:
    polar=train['polar'][index]
    if int(polar) == 0:
        train.drop(index=index,axis=0,inplace=True)

for index in val.index:
    polar =val['polar'][index]
    if int(polar) == 0:
        val.drop(index=index,axis=0,inplace=True)

print(len(train))

train.to_csv('train.csv',index=None)
val.to_csv('val.csv',index=None)
