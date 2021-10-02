import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive/')

mushroom = pd.read_csv('/content/drive/MyDrive/Mushroom edibility classification/mushroom edibility classification dataset.csv')
mushroom.head(12)

mushroom.shape

mushroom.isnull().sum()

mushroom[['cap-shape','cap-color']]

from sklearn.impute import SimpleImputer

impute = SimpleImputer(missing_values=np.nan, strategy='mean')

impute.fit(mushroom[['cap-shape','cap-color']])

mushroom[['cap-shape','cap-color']] = impute.transform(mushroom[['cap-shape','cap-color']])


mushroom.info()

mushroom['bruises'].unique()

mushroom['class'].unique()

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
mushroom['bruises'] = enc.fit_transform(mushroom['bruises'])
mushroom['class'] = enc.fit_transform(mushroom['class'])

print(mushroom[['bruises']].head())
print(mushroom[['class']].head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mushroom[['cap-shape','cap-surface','cap-color','bruises','odor','stalk-shape','stalk-root','stalk-surface-above-ring', 'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']], mushroom['class'],test_size= 0.25, random_state=1)

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))

print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))

X_test_scaled = scaler.transform(X_test)

mushroom.head(12)

#accuracy without scaling
from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier()
knn.fit(X_train, y_train)

print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))

#accuracy with scaling
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)
print("Scaled test set accuracy: {:.2f}".format(
    knn.score(X_test_scaled, y_test)))

feature = mushroom[['cap-shape','cap-surface','cap-color','bruises','odor','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color', 'population','habitat']]
feature

labels = mushroom['class']
labels
