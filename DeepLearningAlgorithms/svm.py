import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri_20220209.csv")

data["class"] = [1 if each == "Abnormal" else 0 for each in data ["class"]]
print(data.head(3))
y = data["class"].values #sınıfları y değişkenine koyma 
x_data = data.drop(["class"], axis=1)  # özellikleri x_data içine koyma

#normalizasyon
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# test ve öğrenme (%15 test, %85 eğitim)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.15, random_state=1)

#Destek vektör makinesi
svm = SVC(random_state=1)
svm.fit(x_train,y_train)

print("destek vektör makinesi test doğruluk : {}".format(svm.score(x_test,y_test)))