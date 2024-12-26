import os
import jieba
import re
import math
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from numpy.linalg import norm

"""此文件用于逆向词汇频率搜索"""

def idf_tf(texts,query):
    sql = []  # 建立检索的数据库
    words = []
    search = {}  # 建立词典
    i = 1  # 表示现在的文档数目
    for temp1 in texts.values():
        words.append(temp1)
        # 先直接进行英文分词
        stemmer = PorterStemmer()
        tokens = word_tokenize(temp1)
        temp1 = [stemmer.stem(word) for word in tokens]
        # 再使用jieba进行中文分词
        # temp1 = jieba.lcut(temp1, cut_all=False)
        tt = len(temp1)
        for p in range(tt):
            tete = temp1[0]  # 每次分词第一个短句，分完后加到temp1的最后，直到分完所有短句为止
            temp1.pop(0)
            tete = re.sub(r'[^\w\s]', '', tete)  # 正则化去掉所有非字符的符号
            tete = jieba.lcut(tete, cut_all=False)
            temp1 = temp1 + tete
        sql.append(temp1)  # 存储分好了词的整个文档
        for word in temp1:
            if word in search.keys():  # 判断索引表中是否已经有该单词存在
                if i in search[word].keys():  # 若存在，在字典对应列表中存入对应的文件，实际上用集合更好
                    search[word][i] += 1
                else:
                    search[word][i] = 1  # 统计单词在该文档出现的次数
            else:
                search[word] = {i: 1}  # 若不存在，新建key并建立对应列表存入文件名,后一个为词频
        i += 1  # i增加，代表下一个文档
    # print(sql)
    # 计算tf_idf向量
    tf_idf = []
    i = 1
    for line in sql:
        temp_id = []
        for temp in search.keys():  # 统计在文档集中出现liy数和在本文档中出现次数
            df = len(search[temp])
            idf = math.log(len(texts) / df + 1)
            if i in search[temp].keys():
                tf = search[temp][i]
            else:
                tf = 0
            temp_idf_tf = tf * idf
            temp_id.append(temp_idf_tf)
        tf_idf.append(temp_id)
        i += 1
    # 对问句分词，计算余弦相似度
    consi = {}
    search_term = re.sub(r'[^\w\s]', '', query)  # 正则化去掉所有标点符号
    search_term = jieba.lcut(search_term, cut_all=False)
    temp_id = []
    for temp in search.keys():  # 统计单词在文档集中出现次数和在本文档中出现次数
        df = len(search[temp])
        idf = math.log(len(texts) / df + 1)
        tf = 0
        for word in search_term:
            if word == temp:
                tf += 1
        temp_idf_tf = tf * idf
        temp_id.append(temp_idf_tf)
    urls = []
    text = []
    for i in texts.keys():
        urls.append(i)
    for i in texts.values():
        text.append(i)
    for j in range(len(texts)):
        A = np.array(temp_id)
        B = np.array(tf_idf[j])
        cosine = np.dot(A, B) / (norm(A) * norm(B))
        if np.isnan(cosine):
            consi[j] = 0
        else:
            consi[j] = cosine  # 计算余弦相似度
    consi = sorted(consi.items(), key=lambda x: x[1], reverse=True)
    print(consi)
    # 存放相似度结果和文本结果
    result = {}
    results = {}
    for i in range(3):
        result[urls[consi[i][0]]] = consi[i][1]
        results[urls[consi[i][0]]] = text[consi[i][0]]
    print(result)
    return results