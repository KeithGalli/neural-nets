import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

# 3
# [0 0 0 0 0 1 0 0 1]


train_df = pd.read_csv('./data/train.csv')
one_hot_color = pd.get_dummies(train_df.color).values
one_hot_marker = pd.get_dummies(train_df.marker).values

labels = np.concatenate((one_hot_color, one_hot_marker), axis=1)
print(labels[0])

print(train_df.head())

model = keras.Sequential([
	keras.layers.Dense(64, input_shape=(2,), activation='relu'),
	keras.layers.Dense(64, activation='relu'),
	keras.layers.Dense(9, activation='sigmoid')])

model.compile(optimizer='adam', 
	          loss=keras.losses.BinaryCrossentropy(from_logits=True),
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

np.random.RandomState(seed=42).shuffle(x)
np.random.RandomState(seed=42).shuffle(labels)

model.fit(x, labels, batch_size=4, epochs=10)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

test_one_hot_color = pd.get_dummies(test_df.color).values
test_one_hot_marker = pd.get_dummies(test_df.marker).values

test_labels = np.concatenate((test_one_hot_color, test_one_hot_marker), axis=1)

print("EVALUATION")
model.evaluate(test_x, test_labels)


print("Prediction", np.round(model.predict(np.array([[0,3], [0,1], [-2, 1]]))))





