# temel bileşen analizi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns = feature_names)
df["sinif"] = y
print(df.head(5))

# PCA

pca = PCA(n_components=2, whiten=True)  # whiten = normalize et
pca.fit(data)

x_pca = pca.transform(data)

print("variance ratio :", pca.explained_variance_ratio_)
print("sum :", sum(pca.explained_variance_ratio_))

# temel bileşenleri görselleştirme
df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1] 

color = ["red", "green", "blue"]
for each in range(3):
    plt.scatter(df.p1[df.sinif == each], df.p2[df.sinif == each],
                 color = color[each], label = iris.target_names[each])

plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()