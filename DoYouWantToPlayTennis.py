#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#1.veri yukleme
veriler = pd.read_csv('THERE IS YOUR DIRECTORY OF DATASET')

print(veriler)
temphum = veriler.iloc[:,1:3].values
#print(temphum)

#encoder: Kategorik -> Numeric
gunler = veriler.iloc[:,0:1].values
#print(gunler)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

gunler[:,0] = le.fit_transform(veriler.iloc[:,0])

#print(gunler)

ohe = preprocessing.OneHotEncoder()
gunler = ohe.fit_transform(gunler).toarray()
#print(gunler)


#encoder: Kategorik -> Numeric
play = veriler.iloc[:,-1:].values
#print(play)


# le dedigimiz labelencoder iste ya!!!
#burada aslinda su kod ile hem play hem windy encode edilebilir...
#veri1 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
#tabi bununla hepsini encode edmis olursun ardindan sadece son iki column secilmeli
#digerlerinin encode edilme sekli analiz icin uygun degil...
play[:,-1] = le.fit_transform(veriler.iloc[:,-1])


#print(play)
#print(type(play))

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=gunler, index = range(14), columns = ['overcast','rainy','sunny'])
#print(sonuc)

sonuc2 = pd.DataFrame(data=temphum, index = range(14), columns = ['temperature','humidity'])
#print(sonuc2)

#yukarida buradaki windy encode islemlerinin kisa yolundan bahsedildi bkz. (play)
windy = veriler.iloc[:,-2:-1].values
#print(windy)
windy = le.fit_transform(veriler.iloc[:,-2])
#print(windy)
#print(type(windy))


sonuc4 = pd.DataFrame(data = windy, index = range(14), columns = ['windy'])
#print(sonuc4)


sonuc3 = pd.DataFrame(data = play[:,:1], index = range(14), columns = ['play'])
#print(sonuc3)


#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
#print(s)

s3=pd.concat([s,sonuc4], axis=1)
#print(s3)
s4 = pd.concat([s3,sonuc3],axis=1)
#print(s4)
temp = s4.filter(['temperature'],axis = 1)
#print(temp)
s2 = s4.drop(columns='temperature')
#print(s2)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,temp,test_size=0.33, random_state=0)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
# print(y_pred)
# print(y_test)

#burada baska sekilde bolmustuk... benim assagida yapacagim daha kolay 
# humidity = s2.iloc[:,3].values
# # print(s2)
# # print(type(humidity))
# sol = s2.iloc[:,:3]
# #print(sol)
# sag = s2.iloc[:,4:]
# #print(sag)

# veri = pd.concat([sol,sag],axis = 1)
# #print(veri)


#benim yaptigim da budur.
humidity = s4.filter(['humidity'],axis=1)
veri = s4.drop(columns = ['humidity'])
# print(humidity)
#print(veri)


x_train, x_test,y_train,y_test = train_test_split(veri,humidity,test_size=0.33, random_state=0)
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)
# print(y_pred)
# print(y_test)
a = (abs(y_test-y_pred)**2)
#print(a)
#print((a.sum())/6)

# #yuzdelerini gorebilmek kullanacagimiz bir kutuphane for more see print func.
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values = veri,axis =1)
#print(X)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values
#print(X_l)
X_l = np.array(X_l,dtype=float)
#humidity bagimli degisken olur burada, onceden bakilirsa X_l aslinda veridir,
model = sm.OLS(humidity,X_l).fit()
#print(model.summary())


# # backward elimination zamani...

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(humidity,X_l).fit()
# print(model.summary())


X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(humidity,X_l).fit()
# print(model.summary())
