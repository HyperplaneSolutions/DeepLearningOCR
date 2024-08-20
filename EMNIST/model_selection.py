import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.layers import LeakyReLU
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class ModelCandidate:
    def __init__(self, name, model, color):
        self.name = name
        self.model = model
        self.color = color
        self.history = []
        self.evaluation = []
        
    def print_performance(self, testY, testYPredicted):
        precision = precision_score(testY, testYPredicted, average='weighted')
        recall = recall_score(testY, testYPredicted, average='weighted')
        print(f"\n{self.name} performance:")
        print(f"Training Accuracy: {get_last_accuracy(self.history)}")
        print(f"Test Accuracy: {self.evaluation[1]}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

def get_models(n_categories):
    tiny_model = Sequential([
        Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(n_categories, activation='softmax')
    ])
    tiny_model.compile(Adam(0.001), loss=keras.losses.CategoricalCrossentropy, metrics=['accuracy'])

    simple_model = Sequential([
        Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(84, activation='relu'),
        Dense(n_categories, activation='softmax')
    ])
    simple_model.compile(Adam(0.001), loss=keras.losses.CategoricalCrossentropy, metrics=['accuracy'])

    optimal_model = Sequential([
        Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='tanh', input_shape=(28, 28, 1)),
        MaxPooling2D(strides=2),
        Conv2D(filters=48, kernel_size=(5,5), padding='same', activation='tanh'),
        MaxPooling2D( strides=2),
        Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='tanh'),
        Flatten(),
        Dense(512, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(n_categories, activation='softmax')
    ])
    optimal_model.compile(Adam(0.001), loss=keras.losses.CategoricalCrossentropy, metrics=['accuracy'])
    
    
    return [
        ModelCandidate("Tiny", tiny_model, "r"),
        ModelCandidate("Simple", simple_model, "b"),
        ModelCandidate("Conv. NN", optimal_model, "g"),
        ]

def get_last_accuracy(history):
    acc_array = history.history['accuracy']
    return acc_array[len(acc_array) - 1]
    