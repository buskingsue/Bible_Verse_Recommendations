import pandas as pd
from konlpy.tag import Okt
import collections
import numpy as np


# 🔹 데이터 파일 경로
bible_file = "./data/merged_bible.csv"  # 성경 데이터 파일
stopwords_file = "./StopWord/stopwords.csv"  # 불용어 파일

# 🔹 CSV 데이터 불러오기
df = pd.read_csv(bible_file)

# 🔹 데이터 전처리 (결측값 제거, 특수문자 제거)
df.dropna(inplace=True)
df['content'] = df['content'].astype(str).str.replace(r'[^가-힣\s]', '', regex=True)

# 🔹 불용어 목록 불러오기
stopwords_df = pd.read_csv(stopwords_file)
stopwords = set(stopwords_df['stopword'].tolist())

# 🔹 형태소 분석기(Okt) 사용
okt = Okt()
df['tokenized'] = df['content'].apply(
    lambda x: [word for word, pos in okt.pos(x, stem=True) if pos in ['Noun', 'Verb', 'Adjective']]
)
df['filtered'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stopwords])  # 불용어 제거
df['processed'] = df['filtered'].apply(lambda x: ' '.join(x))  # 띄어쓰기 결합


# 🔹 불필요한 열 삭제
df.drop(columns=['content', 'filtered'], inplace=True)


# 🔹 CSV 파일 불러오기
df = pd.read_csv('./data/processed_bible.csv')

# 🔹 모든 형태소를 리스트로 변환
all_words = ' '.join(df['processed'].dropna()).split()

# 🔹 형태소별 등장 횟수 계산
word_counts = collections.Counter(all_words)

# 🔹 빈도 리스트 생성
freq_list = list(word_counts.values())

# 🔹 중위값 계산
median_value = np.median(freq_list)

# 🔹 결과 출력
print(f"🔹 형태소 총 개수: {len(word_counts)}")
print(f"🔹 중위값(중앙값) 빈도: {int(median_value)}")

# 🔹 상위 10개 형태소 출력 (등장 횟수 기준)
print("\n🔝 가장 많이 등장한 형태소 10개:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}회")


# 🔹 전처리된 데이터 저장
df.to_csv("./data/processed_bible.csv", index=False, encoding='utf-8-sig')

print("✅ 데이터 전처리 완료! `processed_bible.csv` 저장됨.")

