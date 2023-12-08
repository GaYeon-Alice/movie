import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from settings import file
from model.labeling import encoder, decoder
from model.model import train_model, evaluate_model, predict_all
from model.result import save_result

data = pd.read_csv(file)
raw = data.copy()

# 데이터 전처리
sparse_features = ['userId', 'title', 'genres', 'tag']
data = encoder(data, sparse_features)

# 데이터 분할
train, test = train_test_split(data, test_size=0.2, random_state=1204)

# 모델 훈련
feature_names, model = train_model(data, sparse_features, train)

# 모델 평가
evaluate_model(test, feature_names, model)

# 모델로 전체 데이터에 대한 확률 예측
pred_all = predict_all(data, feature_names, model)

# userId, title, prob로 이루어진 데이터프레임 생성
result = pd.DataFrame()
result = decoder(result, raw, data)

# csv 파일로 저장
save_result(result, pred_all)