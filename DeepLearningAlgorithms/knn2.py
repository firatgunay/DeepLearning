from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


iris_dataset = load_iris()

X_ogren, X_test, Y_ogren, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'])
print(X_ogren.shape)
print(X_test.shape)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_ogren,Y_ogren)

X_yeni = [[3.5,2.1,3.4,1.2]]
tahmin = knn.predict(X_yeni)

dogruluk = knn.predict(X_test)

y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test,y_pred)
print(cm)

acc = accuracy_score(y_pred, Y_test)
print(acc)

#sıcaklık haritası
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="white", fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
