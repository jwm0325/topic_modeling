from openpyxl import load_workbook, Workbook
from itertools import islice
from pandas import DataFrame
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from gensim.models import Phrases, LdaModel
from gensim.corpora import Dictionary

nltk.download('wordnet')

# 엑셀 데이터 파일 로드
wb = load_workbook("pargraph_extracted.xlsx",data_only=True)
ws = wb.active
data = ws.values; cols = next(data); data = list(data)
idx = [r[0] for r in data]
data = (islice(r, 0, None) for r in data)
df = DataFrame(data,columns= cols)
num_data = df.shape[0]
paragraphs = df['contents'].values.tolist()
num_to_print =2
print(paragraphs[:2])

# 텍스트 데이터 전처리 시작
# 1. 단어 단위로 토큰화
tokenizer = RegexpTokenizer(r'\w+')
for j in range(len(paragraphs)):
    paragraphs[j] = tokenizer.tokenize(paragraphs[j])
print("(1) Regular Expression을 활용해 특수문자를 제거, 단어 단위 구분")
print(paragraphs[:num_to_print])

#2. 불용어 제거
stop_words = set(stopwords.words('english'))
# 필요시 추가
custom_stopwords = {}
stop_words = stop_words.union(custom_stopwords)
paragraphs = [[token for token in p if token not in stop_words] for p in paragraphs]
print("\n(2) 전체 단어 토근에서 불용어(Stopwords) 제거")
print(paragraphs[:num_to_print])

paragraphs = [[token for token in p if not token.isnumeric()] for p in paragraphs]
print("\n(3) 숫자 토큰 제거")
print(paragraphs[:num_to_print])

paragraphs = [[token for token in p if len(token)> 1] for p in paragraphs]
print("\n(4) 한글자(단일 문자) 제거")
print(paragraphs[:num_to_print])

# 5. 표제어 추출 ( ex: cars -> car )
lemmatizer = WordNetLemmatizer()
paragraphs = [[lemmatizer.lemmatize(token) for token in p] for p in paragraphs]
print("\n(5) 표제어 추출 (word의 개수를 낮추는 용도)")
print(paragraphs[:num_to_print])

# 6 Bigram (연속된 2 단어를 하나로)
bigram = Phrases(paragraphs,min_count=20)
for j in range(len(paragraphs)):
    for token in bigram[paragraphs[j]]:
        if '-' in token: paragraphs[j].append(token)
print("\n(6) Bigram (연속된 두 단어를 하나도)")
print(paragraphs[:num_to_print])

dictionary = Dictionary(paragraphs)
print(f"\n단어의 수: {len(dictionary)}")

# with open('dict_raw.txt','w',encoding='utf-8') as f:
#     for k,v in dictionary.token2id.items(): f.write("%s %s\n" % (v,k))

dictionary.filter_extremes(no_below=20, no_above=1.0)
print(f"\n필터링 후 단어의 수: {len(dictionary)}")

with open('dict_filtered.txt','w',encoding='utf-8') as f:
    for k, v in dictionary.token2id.items(): f.write("%s %s\n" % (v,k))

# 9. Bow 코퍼스 생성
corpus = [dictionary.doc2bow(p) for p in paragraphs]
print(f"\n말뭉치의 수 : {len(corpus)}")
print(corpus[:num_to_print])

num_topics = 6 # 마음대로 수정 가능
chunk_size = 2000 # 한 번에 학습하는 데이터의 양
passes = 20 # 전체 코퍼스 반복 횟수
iteration = 400 # 한 패스에서 문서를 반복하는 횟수
eval_every = None

#11 토픽 모델링 실행
temp =dictionary[0]
id2word = dictionary.id2token
model = LdaModel(
    corpus = corpus,
    id2word= id2word,
    chunksize= chunk_size,
    alpha= 'auto',
    eta = 'auto',
    iterations= iteration,
    num_topics = num_topics,
    passes= passes,
    eval_every = eval_every
)
top_topics = model.top_topics(corpus)
from pprint import pprint
pprint(top_topics)
model.save('./lda_topic_6.model')

#12 시각화
import pyLDAvis
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False, R=10)
pyLDAvis.save_html(vis, './lda_6.html')

#13 결과 저장
top_words_per_topics = []
for t in range(model.num_topics):
    top_words_per_topics.extend([(t,) + x for x in model.show_topic(t,topn=20)])
import pandas as pd
pd.DataFrame(
    top_words_per_topics, columns = ['Topic','Word','P']
).to_csv('top_word.csv')