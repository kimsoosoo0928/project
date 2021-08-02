# 시작에 앞서 한글 폰트설정

def get_font_family():
    """
    시스템 환경에 따른 기본 폰트명을 반환하는 함수
    """
    import platform
    system_name = platform.system()
    # colab 사용자는 system_name이 'Linux'로 확인
    
    if system_name == "Darwin" :
        font_family = "AppleGothic"
    elif system_name == "Windows" :
        font_family = "Malgun Gothic"
    else:
        # Linux
        # colab에서는 runtime을 꼭 재시작 해야합니다.
        # 런타임을 재시작 하지 않고 폰트 설치를 하면 기본 설정 폰트가 로드되어 한글이 깨집니다.
        #!apt-get update -qq
        #!apt-get install fonts-nanum -qq > /dev/null
        
        import matplotlib.font_manager as fm
        
        fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        font = fm.FontProperties(fname=fontpath, size=9)
        fm._rebuild()
        font_family = "NanumBarunGothic"
        
    return font_family

# https://www.youtube.com/watch?v=9ovF2bqMME4
# 유튜버 "todaycode오늘코드"님의 영상 < KRX분석 [6/13] 데이터 시각화 도구 소개와 한글 폰트 설정 >
# 데이콘에서는 "Visualising Korea"로 활동하시는 듯 합니다.

import itertools
import warnings

import numpy as np
import pandas as pd

import math
from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

font_family = get_font_family()
plt.rc("font", family=font_family)
path = './_data/energy/'

train = pd.read_csv(path + 'train.csv', encoding='CP949')
test = pd.read_csv(path + 'test.csv', encoding='CP949')
submission = pd.read_csv(path + 'sample_submission.csv')

train['date_time'] = pd.to_datetime(train['date_time'],format='%Y-%m-%d %H')
train.set_index('date_time', drop=True, inplace=True)

test['date_time'] = pd.to_datetime(test['date_time'],format='%Y-%m-%d %H')
test.set_index('date_time', drop=True, inplace=True)

# 자료
print("자료 : ", train.columns.tolist())

# 기간
print("기간 : 분석기간(", train.index[0], "~", train.index[train.shape[0]-1], "), 예측기간(", test.index[0], "~", test.index[test.shape[0]-1], ")")

# 분석내용
print("분석내용 : 기간, 기상요소 반영 전후의 전국 전력사용량 예측 오차 분석")

# https://rfriend.tistory.com/494
# UDF of Resampling by column name, time span and summary functions


def resample_summary(ts_data, col_nm, time_span, func_list):
    
    df_summary = pd.DataFrame() # blank DataFrame to store results

    # resampler with column name by time span (group by)
    resampler = ts_data[col_nm].resample(time_span)

    # aggregation functions with suffix name
    if 'first' in func_list:
        df_summary[col_nm + '_' + time_span + '_first'] = resampler.first()

    if 'last' in func_list:
        df_summary[col_nm + '_' + time_span + '_last'] = resampler.last()

    if 'sum' in func_list:
        df_summary[col_nm + '_' + time_span + '_sum'] = resampler.sum()

    if 'cumsum' in func_list:
        df_summary[col_nm + '_' + time_span + '_cumsum'] = resampler.sum().cumsum()

    if 'min' in func_list:
        df_summary[col_nm + '_' + time_span + '_min'] = resampler.min()

    if 'max' in func_list:
        df_summary[col_nm + '_' + time_span + '_max'] = resampler.max()

    if 'mean' in func_list:
        df_summary[col_nm + '_' + time_span + '_mean'] = resampler.mean()

    if 'median' in func_list:
        df_summary[col_nm + '_' + time_span + '_median'] = resampler.median()

    if 'range' in func_list:
        df_summary[col_nm + '_' + time_span + '_range'] = resampler.max() - resampler.min()

    if 'var' in func_list:
        df_summary[col_nm + '_' + time_span + '_var'] = resampler.var() # sample variance

    if 'stddev' in func_list:
        df_summary[col_nm + '_' + time_span + '_stddev'] = np.sqrt(resampler.var())

    return df_summary

func_list = ['mean', 'min', 'max', 'var', '_stddev']
rs = resample_summary(train, '전력사용량(kWh)', '1D', func_list)
# ts_data = train
# col_nm : 전력사용량(kWh)
# time_span : 1D *하루단위


# 일 별 평균 전력사용량
plt.figure(figsize=(15, 5))
plt.title('평균 전력 사용량')
rs['전력사용량(kWh)_1D_mean'].plot()

plt.show()

################################################################

train = pd.read_csv(path + 'train.csv', encoding='CP949')
test = pd.read_csv(path + 'test.csv', encoding='CP949')
submission = pd.read_csv(path + 'sample_submission.csv')

train['비전기냉방설비운영'].fillna(0, inplace=True)
train['태양광보유'].fillna(0, inplace=True)

# 시각화를 위해 split 해줍니다.
val = train.query('"2020-08-18" <= date_time < "2020-08-25"')
train = train.query('"2020-06-01" <= date_time < "2020-08-18"')

#2d의 데이터프레임을 건물별 정보를 반영한 3d 데이터로 변환
def df2d_to_array3d(df_2d):
    feature_size=df_2d.iloc[:,2:].shape[1]
    time_size=len(df_2d['date_time'].value_counts())
    sample_size=len(df_2d.num.value_counts())
    return df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])

# 전력소비량만을 사용했을 경우

train_x_array=df2d_to_array3d(train)
test_x_array=df2d_to_array3d(val)

print(train_x_array.shape) # (60, 1872, 8)
print(test_x_array.shape) # (60, 168, 8)

idx=1
x_series=train_x_array[idx, :, 0]
model=ARIMA(x_series, order=(3, 0, 1))
fit=model.fit()

preds=fit.predict(1, 168, typ='levels')

# 예측 시각화
plt.figure(figsize=(10, 5))
plt.plot(x_series, label = 'input_series')
plt.plot(np.arange(1872, 1872+168), test_x_array[idx, :, 0], label='y')
plt.plot(np.arange(1872, 1872+168), preds, label='prediction')
plt.legend()

plt.show()

# 확대 보기
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1872, 1872+168), test_x_array[idx, :, 0], label='y')
plt.plot(np.arange(1872, 1872+168), preds, label='prediction')
plt.legend()
plt.show()

#########################################################################
# 기간 세분화 & 기상요소 다양화 반영을 하였을 경우
# 체감온도와 불쾌지수 파생변수 생성: 불쾌지수는 간단한 식이 있어 이를 이용하였다.
# https://www.weather.go.kr/plus/life/li_asset/HELP/basic/help_01_07.jsp
# http://www.psychiatricnews.net/news/articleView.html?idxno=10116

def get_pow(series):
    return math.pow(series, 0.15)

train['perceived_temperature'] = 13.12 + 0.6215*train['기온(°C)'] - 11.37*train['풍속(m/s)'].apply(get_pow) + 0.3965*train['풍속(m/s)'].apply(get_pow)*train['기온(°C)']
train['discomfort_index'] = 1.8*train['기온(°C)'] - 0.55*(1-train['습도(%)'])*(1.8*train['기온(°C)']-26) + 32

val['perceived_temperature'] = 13.12 + 0.6215*val['기온(°C)'] - 11.37*val['풍속(m/s)'].apply(get_pow) + 0.3965*val['풍속(m/s)'].apply(get_pow)*val['기온(°C)']
val['discomfort_index'] = 1.8*val['기온(°C)'] - 0.55*(1-val['습도(%)'])*(1.8*val['기온(°C)']-26) + 32

# 날짜에 대한 변수 생성: 월, 요일

train['month'] = pd.to_datetime(train.date_time).dt.month
train['weekday'] = pd.to_datetime(train.date_time).dt.weekday
train['hour'] = pd.to_datetime(train.date_time).dt.hour

val['month'] = pd.to_datetime(val.date_time).dt.month
val['weekday'] = pd.to_datetime(val.date_time).dt.weekday
val['hour'] = pd.to_datetime(val.date_time).dt.hour

def weekend(day):
    if day >= 5:
        return 1
    else:
        return 0

train['weekday'] = train['weekday'].apply(lambda x: weekend(x))
val['weekday'] = val['weekday'].apply(lambda x: weekend(x))

# train = train[['num', 'date_time', '전력사용량(kWh)', 'perceived_temperature', 'discomfort_index', 'weekday', 'hour']]
# val = val[['num', 'date_time', '전력사용량(kWh)', 'perceived_temperature', 'discomfort_index', 'weekday', 'hour']]

train_x_array=df2d_to_array3d(train)
test_x_array=df2d_to_array3d(val)

print(train_x_array.shape)
print(test_x_array.shape)

idx=1

x_series=train_x_array[idx, :, 0]
x_else = train_x_array[idx, :, 1:]
val_else = test_x_array[idx, :, 1:]



mod = sm.tsa.statespace.SARIMAX(x_series,
                                x_else,
                                order=(3, 0, 1),
                                seasonal_order=(0, 0, 0, 0))
results = mod.fit()

print(results.summary().tables[1])

pred = results.predict(start = 1872, end = 2039, exog=val_else, dynamic= True)

# 예측 결과 시각화
plt.figure(figsize=(10, 5))
plt.plot(x_series, label = 'input_series')
plt.plot(np.arange(1872, 1872+168), test_x_array[idx, :, 0], label='y')
plt.plot(np.arange(1872, 1872+168), pred, label='prediction')
plt.legend()
plt.show()


# 더 자세하게 보면 다음과 같습니다.
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1872, 1872+168), test_x_array[idx, :, 0], label='y')
plt.plot(np.arange(1872, 1872+168), pred, label='prediction')
plt.legend()
plt.show()
