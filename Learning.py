import os
import numpy as np
import pandas as pd

#Считывание набора данных
def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))

df = read_data('/data/notebook_files', 'train.csv')
df.head()

#Сохранение данных под переменными, Х - признаки, у - метки для обучения или проверки модели
def load_dataset(label_dict):
    train_X = read_data('/data/notebook_files', 'train.csv').values[:,:-2]
    train_y = read_data('/data/notebook_files', 'train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data('/data/notebook_files', 'test.csv').values[:,:-2]
    test_y = read_data('/data/notebook_files', 'test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return(train_X, train_y, test_X, test_y)
label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}
train_X, train_y, test_X, test_y = load_dataset(label_dict)

#Выбор модели
from sklearn.neighbors import KNeighborsClassifier as mymod
#Инициализация модели
model = mymod()

#Обучение модели
model.fit(train_X, train_y)

#Оценка модели
yhat = model.predict(test_X)

from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

print(classification_report(test_y, yhat, target_names=target_names))
