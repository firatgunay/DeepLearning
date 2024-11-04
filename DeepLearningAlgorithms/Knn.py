import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri_20220209.csv")
#print(data.head(3))

sns.scatterplot(data=data, x="lumbar_lordosis_angle", y="pelvic_tilt numeric", hue="class")
plt.xlabel("lomber lordoz açısı")
plt.ylabel("pelvik eğim")
plt.legend()

data["class"] = [1 if each == "Abnormal" else 0 for each in data ["class"]]
print(data.head(3))
y = data["class"].values #sınıfları y değişkenine koyma 
x_data = data.drop(["class"], axis=1)  # özellikleri x_data içine koyma

#normalizasyon

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# test ve öğrenme (%15 test, %85 eğitim)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.15, random_state=1)

#knn model
komsu_sayisi=4
knn = KNeighborsClassifier(n_neighbors=komsu_sayisi)
knn.fit(x_train,y_train)

predict = knn.predict(x_test)
print("{} en yakın komşu modeli test doğruluk : {}".format(komsu_sayisi, knn.score(x_test,y_test)))

#en iyi k değeri bulma

score_list = []
for each in range(1,50):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,50), score_list)
plt.xlabel("k değerleri")
plt.ylabel("doğruluk")
plt.title("en iyi k değerlerin bulunması")



# confusion matrix
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

#sıcaklık haritası
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white", fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
#plt.show()