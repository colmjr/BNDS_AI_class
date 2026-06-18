# -*- coding: utf-8 -*-

import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
# 添加黑体字体路径
font_path = '/System/Library/Fonts/STHeiti Light.ttc'
prop = fm.FontProperties(fname=font_path)

# 使用字体属性绘制图形
plt.rcParams['font.family'] = prop.get_name()

characters = [
    "黛玉", "宝钗", "贾元春", "贾迎春", "贾探春", "贾惜春", "李纨", "妙玉", "史湘云", "王熙凤", "贾巧姐", "秦可卿",
    "晴雯", "麝月", "袭人", "鸳鸯", "雪雁", "紫鹃", "碧痕", "平儿", "香菱", "金钏", "司棋", "抱琴",
    "赖大", "焦大", "王善保", "周瑞", "林之孝", "乌进孝", "包勇", "吴贵", "吴新登", "邓好时", "王柱儿", "余信",
    "庆儿", "昭儿", "兴儿", "隆儿", "坠儿", "喜儿", "寿儿", "丰儿", "住儿", "小舍儿", "李十儿", "玉柱儿",
    "贾敬", "贾赦", "贾政", "宝玉", "贾琏", "贾珍", "贾环", "贾蓉", "贾兰", "贾芸", "贾蔷", "贾芹",
    "琪官", "芳官", "藕官", "蕊官", "药官", "玉官", "宝官", "龄官", "茄官", "艾官", "豆官", "葵官",
    "妙玉", "智能", "智通", "智善", "圆信", "大色空", "净虚",
    "彩屏", "彩儿", "彩凤", "彩霞", "彩鸾", "彩明", "彩云",
    "贾元春", "贾迎春", "贾探春", "贾惜春",
    "宝玉", "甄宝玉", "薛宝钗", "薛宝琴",
    "薛蟠", "薛蝌", "宝钗", "薛宝琴",
    "王夫人", "王熙凤", "王子腾", "王仁",
    "尤老娘", "尤氏", "尤二姐", "尤三姐",
    "贾蓉", "贾兰", "贾芸", "贾蔷",
    "贾珍", "贾琏", "贾环", "贾瑞",
    "贾敬", "贾赦", "贾政", "贾敏",
    "贾代儒", "贾代化", "贾代修", "贾代善",
    "晴雯", "金钏", "鸳鸯", "司棋",
    "秦锺", "蒋玉菡", "柳湘莲", "东平王",
    "乌进孝", "冷子兴", "山子野", "方椿",
    "载权", "夏秉忠", "周太监", "裘世安",
    "抱琴", "司棋", "侍画", "入画",
    "珍珠", "琥珀", "玻璃", "翡翠",
    "史湘云", "翠缕", "笑儿", "篆儿",
    "贾探春", "侍画", "翠墨", "小蝉",
    "贾宝玉", "茗烟", "袭人", "晴雯",
    "林黛玉", "紫鹃", "雪雁", "春纤",
    "贾惜春", "入画", "彩屏", "彩儿",
    "贾迎春", "彩凤", "彩云", "彩霞"
]

unique_characters = list(set(characters))
for word in unique_characters:
    jieba.suggest_freq(word, True)

# 读取文本文件
f = open("hongloumeng.txt", 'r', encoding='utf-8')  # 读入文本
lines = []
for line in f:  # 分别对每段分词
    temp = jieba.lcut(line)  # jieba分词
    words = []
    for i in temp:
        # 过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》]+|[+——！，。？、~@#￥%……&*（）：；‘1234567890—]", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])  # 预览前5行分词结果

# 调用Word2Vec训练
model = Word2Vec(lines, vector_size=20, window=3, min_count=3, epochs=20, negative=10, sg=1)
print("宝玉的词向量：\n", model.wv.get_vector('明知'))
print("\n和“宝玉”相关性最高的词语：")
print(model.wv.most_similar('明知', topn=20))  # 与宝玉最相关的前20个词语

# 将词向量投影到二维空间
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key):  # index_to_key 序号,词语
    rawWordVec.append(model.wv[w])  # 词向量
    word2ind[w] = i  # {词语:序号}
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

# 绘制星空图
fig = plt.figure(figsize=(9, 6))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')


# 绘制几个特殊单词的向量
words = ['宝玉', '宝钗', '黛玉', '袭人', '大观园', '林妹妹', '宝姐姐', '凤丫头', '宁国府', '荣国府']
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha=1, color='red', markersize=12)
        plt.text(xy[0], xy[1], w, alpha=1, fontsize=18, color='blue')

plt.show()