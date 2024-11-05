import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri_20220209.csv")
print(data.head(5))

sns.countplot(data["class"])

data["class"] = [1 if each == "Abnormal" else 0 for each in data ["class"]]
#data.info()

y = data["class"].values #sınıfları y değişkenine koyma 
x_data = data.drop(["class"], axis=1)  # özellikleri x_data içine koyma
sns.pairplot(x_data)
#plt.show()


#normalizasyon

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# test ve öğrenme (%15 test, %85 eğitim)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.15, random_state=42)

# transpose
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train : ", x_train.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)

#eğitim
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)

#test
test_dogrulugu = lr.score(x_test.T, y_test.T)*100
print("test doğruluğu : {}".format(test_dogrulugu))