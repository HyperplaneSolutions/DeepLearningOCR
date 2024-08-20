import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def show_image(df, encoding, index):
    df_img_only = df.iloc[: , 1:]
    imageRow = df_img_only.iloc[index].to_numpy()
    imageRowNp = np.array(imageRow, dtype='float')
    pixels = imageRowNp.reshape((28, 28)).transpose()
    
    codeRaw = df.iloc[index].to_numpy()[0]
    anscii = encoding.loc[encoding['I'] == codeRaw]['ANSCII']

    actual = chr(int(anscii.iloc[0]))
    plt.title(actual)

    plt.imshow(pixels, cmap='gray')
    plt.show()

def plot_histories(model_candidates):
    plt.figure()
    for x in model_candidates:
        plt.plot(x.history.history['accuracy'], label=x.name, color = x.color)
    
    plt.title('Accuracy per model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
   