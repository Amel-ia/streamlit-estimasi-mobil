import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('merc.csv')

#Deskripsi Dataset

df.head()

df.info()

sns.heatmap(df.isnull())

df.describe()

#visualisasi data
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)

#jumlah mobil berdasarkan model 
models = df.groupby('model').count()[['tax']].sort_values(by='tax',ascending=True).reset_index()
models = models.rename(columns={'tax':'numberOfCars'})

fig = plt.figure(figsize=(15,5))
sns.barplot(x=models['model'], y=models['numberOfCars'], color='pink')
plt.xticks(rotation=60)

#ukuran mesin

engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax').reset_index()
engine = engine.rename(columns={'tax':'count'})

plt.figure(figsize=(15,5))
sns.barplot(x=engine['engineSize'], y=engine['count'],color='royalblue')


#distribusi mileage
plt.figure(figsize=(15,5))
sns.displot(df['mileage'])

#distribusi harga mobil
plt.figure(figsize=(15,5))
sns.displot(df['price'])


#seleksi fitur
features = ['year','mileage','tax','mpg','engineSize']
x = df[features]
y = df  ['price']
x.shape, y.shape

#split data training dan data testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape


#membuat model regresi linier

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)

score = lr.score(x_test, y_test)
print('akurasi model regresi linier = ', score)

#membuat inputan model regresi linier
#year= 2019, mileage=5700, tax=135, mpg=31.3, engineSize=2
input_data = np.array([[2019,5700,135,31.3,2]])

prediction = lr.predict(input_data)
print('Estimasi harga mobil dalam EUR :', prediction)

import pickle

filename = 'Estimasi_Mobil.sav'
pickle.dump(lr,open(filename,'wb'))



