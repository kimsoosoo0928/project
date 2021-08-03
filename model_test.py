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
        # !apt-get update -qq
        # !apt-get install fonts-nanum -qq > /dev/null
        
        import matplotlib.font_manager as fm
        
        fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        font = fm.FontProperties(fname=fontpath, size=9)
        fm._rebuild()
        font_family = "NanumBarunGothic"
        
    return font_family

### import ### 

import itertools
import warnings

import numpy as np
import pandas as pd

import math
from tqdm.notebook import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


font_family = get_font_family()
plt.rc("font", family=font_family)
path = './_data/energy/'

### data ###

train = pd.read_csv(path + 'train.csv', encoding='CP949')
test = pd.read_csv(path + 'test.csv', encoding='CP949')
submission = pd.read_csv(path + 'sample_submission.csv')

train['date_time'] = pd.to_datetime(train['date_time'],format='%Y-%m-%d %H')
train.set_index('date_time', drop=True, inplace=True)

test['date_time'] = pd.to_datetime(test['date_time'],format='%Y-%m-%d %H')
test.set_index('date_time', drop=True, inplace=True)

# 자료
print("자료 : ", train.columns.tolist())
# 자료 :  ['num', '전력사용량(kWh)', '기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', '일조(hr)', '비전기냉방설비운영', '태양광보유']

# 기간
print("기간 : 분석기간(", train.index[0], "~", train.index[train.shape[0]-1], "), 예측기간(", test.index[0], "~", test.index[test.shape[0]-1], ")")
# 기간 : 분석기간( 2020-06-01 00:00:00 ~ 2020-08-24 23:00:00 ), 예측기간( 2020-08-25 00:00:00 ~ 2020-08-31 23:00:00 )

### 시계열 데이터를 특정 시간 단위 구간별로 요약 통계량 구하는 사용자 정의 함수 ###

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

train = pd.read_csv(path + 'train.csv', encoding='CP949')
test = pd.read_csv(path + 'test.csv', encoding='CP949')
submission = pd.read_csv(path + 'sample_submission.csv')

# 결측치 0으로 대체
train['비전기냉방설비운영'].fillna(0, inplace=True)
train['태양광보유'].fillna(0, inplace=True)


# 시각화를 위해 split 해준다.
val = train.query('"2020-08-18" <= date_time < "2020-08-25"')
train = train.query('"2020-06-01" <= date_time < "2020-08-18"')

#2d의 데이터프레임을 건물별 정보를 반영한 3d 데이터로 변환
def df2d_to_array3d(df_2d):
    feature_size=df_2d.iloc[:,2:].shape[1]
    time_size=len(df_2d['date_time'].value_counts())
    sample_size=len(df_2d.num.value_counts())
    return df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])


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
                                order=(0,0,0),
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

print('기간 세분화 & 기상요소 다양화 반영 mae : ',mean_absolute_error(test_x_array[idx, :, 0], pred)) # 216.42659428701504
print('기간 세분화 & 기상요소 다양화 반영 R2 : ', r2_score(test_x_array[idx, :, 0], pred)) # 0.6040675014510193

'''
order=(0,0,0)
seasonal_order=(0, 0, 0, 0))
기간 세분화 & 기상요소 다양화 반영 mae :  199.7081345946761
기간 세분화 & 기상요소 다양화 반영 R2 :  0.7101115801081498

order=(0,0,1),
seasonal_order=(0, 0, 0, 0))
기간 세분화 & 기상요소 다양화 반영 mae :  203.21908184161188
기간 세분화 & 기상요소 다양화 반영 R2 :  0.6924044531955753

order=(0,0,2),
seasonal_order=(0, 0, 0, 0))
기간 세분화 & 기상요소 다양화 반영 mae :  205.01772609783083
기간 세분화 & 기상요소 다양화 반영 R2 :  0.6830923948492192

order=(0,0,3),
seasonal_order=(0, 0, 0, 0))
기간 세분화 & 기상요소 다양화 반영 mae :  209.3340015131347
기간 세분화 & 기상요소 다양화 반영 R2 :  0.6555804977218336

order=(0,1,0),
seasonal_order=(0, 0, 0, 0))
기간 세분화 & 기상요소 다양화 반영 mae :  262.3296264847448
기간 세분화 & 기상요소 다양화 반영 R2 :  0.1160467576672386

order=(1,0,0),
seasonal_order=(0, 0, 0, 0))
기간 세분화 & 기상요소 다양화 반영 mae :  339.92217384163166
기간 세분화 & 기상요소 다양화 반영 R2 :  0.19663481002351535

order=(0,0,0),
seasonal_order=(2, 2, 12, 2))
기간 세분화 & 기상요소 다양화 반영 mae :  316.31162499831424
기간 세분화 & 기상요소 다양화 반영 R2 :  -0.08874561759491284

order=(0,0,0),
seasonal_order=(2, 2, 2, 2))
기간 세분화 & 기상요소 다양화 반영 mae :  277.6379056405458
기간 세분화 & 기상요소 다양화 반영 R2 :  0.010451429068208062
'''