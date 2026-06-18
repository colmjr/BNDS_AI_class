from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
# 1. 准备数据：一个简单的语料库
sentences = [
["我爱","学习","自然语言处理"],
["Word2Vec","是","一个","强大","的","工具"],
["深度学习","推动","了","NLP","的","发展"]
]
# 2. 训练Word2Vec模型
model = Word2Vec(
sentences,# 训练数据
vector_size=100,# 词向量维度
window=5,# 上下文窗口大小
min_count=1,# 忽略出现次数少于1的词
sg=0,# 0=CBOW, 1=Skip-gram
workers=4# 线程数
)
# 3. 使用模型：获取词向量、计算相似度、语义类比
vector = model.wv['学习']
print(f"词'学习'的向量前5个维度: {vector[:5]}")
similarity = model.wv.similarity('学习','发展')
print(f"'学习'和'发展'的相似度: {similarity}")
