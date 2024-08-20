import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import keras
from model_selection import get_models

from visualization import plot_histories, show_image

from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.layers import LeakyReLU
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_ROW_LIMIT = 9999999999
TEST_ROW_LIMIT = 9999999999

train_df_raw = pd.read_csv("./dataset/EMNIST/emnist-balanced-train.csv", nrows=TRAIN_ROW_LIMIT)
test_df_raw = pd.read_csv("./dataset/EMNIST/emnist-balanced-test.csv", nrows=TEST_ROW_LIMIT)

encoding = pd.read_csv("./dataset/EMNIST/emnist-balanced-mapping.txt", header=None, delimiter=r"\s+", names=['I', 'ANSCII'])
n_categories = int(encoding.count()[0])

X_train = train_df_raw.iloc[: , 1:].to_numpy(dtype=np.float32)
y_train = train_df_raw.iloc[: , 0].to_numpy()

X_test = test_df_raw.iloc[: , 1:].to_numpy(dtype=np.float32)
y_test = test_df_raw.iloc[: , 0].to_numpy()

# normalize
X_train /= 255
X_test /= 255

# onehot
y_train = keras.utils.to_categorical(y_train, n_categories)
y_test = keras.utils.to_categorical(y_test, n_categories)

# reshape for keras
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(X_train.shape)

# calculate mean and standard deviation
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

# function to normalize input data
def norm_input(x): return (x-mean_px)/std_px

model_candidates = get_models(n_categories)

training_epochs = 1
batch_size = 256

for x in model_candidates:
    model = x.model
    history = model.fit(x = X_train,y = y_train,batch_size = batch_size,epochs = training_epochs)
    evaluation = model.evaluate(x = X_test, y = y_test, batch_size = batch_size)
    x.history = history
    x.evaluation = evaluation

print("Model Evaluations:")
for x in model_candidates:
    y_test_predicted = x.model.predict(X_test, verbose=0)
    y_test_predicted_onehot = to_categorical(np.argmax(y_test_predicted, axis=1), n_categories)
    x.print_performance(y_test, y_test_predicted_onehot)

# visualizing accuracy
plot_histories(model_candidates)