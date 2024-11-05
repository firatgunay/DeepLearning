import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix



data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri_20220209.csv")

data["class"] = [1 if each == "Abnormal" else 0 for each in data ["class"]]
print(data.head(3))
y = data["class"].values #sınıfları y değişkenine koyma 
x_data = data.drop(["class"], axis=1)  # özellikleri x_data içine koyma

#normalizasyon
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# test ve öğrenme (%15 test, %85 eğitim)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.15, random_state=1)

#Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=1) # 100 tane ağaç için
rf.fit(x_train,y_train)

print("Random Forest Doğruluk Değeri : {}".format(rf.score(x_test, y_test)))

#confusion matrix
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#sıcaklık haritası
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white", fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()