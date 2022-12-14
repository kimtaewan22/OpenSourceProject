import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np



df = pd.read_csv('people.csv', encoding='cp949', header=None)
df2 = pd.read_csv('출생률.csv',encoding='cp949', header=None)

from matplotlib import font_manager, rc
font_path = "malgun.ttf"   #폰트파일의 위치
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)



df.columns = ['year','date','applicant','examinee','Examination rate','registered student','Graduate','GED','pre to rate']
df.drop(['date'], axis=1 , inplace= True) # 수능 응시 날짜는 삭제

ndf1 = df[['examinee','applicant']]

applicantMean = ( int(df.iloc[0,1]) + int(df.iloc[1,1]) ) // 2
examineeMean = ( int(df.iloc[0,2]) + int(df.iloc[1,2]) ) // 2
rateMean = ( float(df.iloc[0,3]) + float(df.iloc[1,3]) ) / 2

df.drop(index=0, axis=0, inplace=True) # df에서 1~2행 삭제. 
df.set_index('year', inplace=True) 
df.loc[1994,['applicant','examinee','Examination rate']] = applicantMean, examineeMean, rateMean # 1994년의 행 값을 두 번의 평균값으로 대체.



df2 = df2.T
df2.columns = ['year','birth rate']
ndf2 = df2[['birth rate']]
df2.set_index('year', inplace=True)


# print(ndf1.describe())
# print(ndf2.describe())


conNdf = pd.concat([ndf1,ndf2], axis=1)

grouped = df.groupby(['pre to rate'])
group1 = grouped.get_group('up')
group2 = grouped.get_group('down')

aggGroup1 = (group1.agg({'examinee': 'mean'}))
aggGroup2 = (group2.agg({'examinee': 'mean'}))

aggGroup3 = (group1.agg({'Examination rate' : 'mean'}))
aggGroup4 = (group2.agg({'Examination rate' : 'mean'}))

concatGroup1 = (pd.concat([aggGroup1,aggGroup2]))
concatGroup1.index = ['examinee-up','examinee-down']


concatGroup2 = (pd.concat([aggGroup3,aggGroup4]))
concatGroup2.index = ['Examination rate-up','Examination rate-down']

group1Index = str(group1.index)
group2Index = list(group2.index)

x = conNdf[['birth rate']]
y = conNdf[['examinee']]


x_train, x_test, y_train, y_test = train_test_split(x,  
                                                    y,
                                                    test_size=0.3,
                                                    random_state=10)

print('train data count: ', len(x_train))
print('test data count: ', len(x_test))

lr = LinearRegression()
lr.fit(x_train, y_train)

print('coefficent a: ', lr.coef_) # 독립변수의 계수를 출력
print('\n')
print('y intercept b', lr.intercept_) # 절편을 출력

y_hat = lr.predict(x)

Y_pred = lr.predict(x_test) # 검증 데이터를 사용해 종속변수를 예측
Y_train_pred = lr.predict(x_train) # 학습데이터에 대한 종속변수를 예측
print('MSE train data: ', np.sqrt(mean_squared_error(y_train, Y_train_pred))) # 학습 데이터를 사용했을 때의 평균 제곱 오차를 출력
print('MSE test data: ', np.sqrt(mean_squared_error(y_test, Y_pred)))         # 검증 데이터를 사용했을 때의 평균 제곱 오차를 출력



plt.style.use('ggplot')
fig = plt.figure(figsize=(14,10))
fig.subplots_adjust(hspace=1)



plt.subplot(4, 2, 1)
plt.plot(df.index, df.loc[:,'applicant'], marker= 'o', markersize=5, label = "applicant")
plt.plot(df.index, df.loc[:,'examinee'], marker= 'o', markersize=5, label = "examinee")

# ax1.plot(df.index, df.loc[:,'Examination rate'], marker= 'o', markersize=10, label = "Examination rate")
plt.ylabel('지원자 및 응시자')
plt.xticks(df.index, rotation = 60)
plt.legend(loc = 'best')

plt.subplot(4, 2, 3)

plt.plot(df2.index, df2.values, marker= 'o', markersize=5, label = "birth rate")
plt.legend(loc = 'best')
plt.xticks(df2.index,rotation = 60)
plt.ylabel('출생률')
plt.xlabel('연도')

plt.subplot(4, 2, 2)

plt.bar(concatGroup1.index, concatGroup1.values, label = "시험 응시자 평균", width=0.2)
plt.legend(loc = 'best')
plt.ylabel('시험 응시자')
plt.xlabel('up-down')

plt.subplot(4, 2, 4)

plt.bar(concatGroup2.index, concatGroup2.values, label = "시험 응시율 평균", width=0.2)
plt.legend(loc = 'best')
plt.ylabel('시험 응시율')
plt.xlabel('up-down')

plt.subplot(4, 2, 5)

plt.scatter(conNdf['birth rate'],conNdf['examinee'], label = "시험 응시율 평균")
plt.legend(loc = 'best')
plt.ylabel('시험 응시자 수')
plt.xlabel('출생률')

plt.subplot(4, 2, 6)

plt.scatter(conNdf['birth rate'],conNdf['examinee'], label = "시험 응시율 평균")
plt.plot(conNdf['birth rate'], y_hat, color = 'red')
plt.legend(loc = 'best')
plt.ylabel('y_hat')
plt.xlabel('출생률')

plt.show()
