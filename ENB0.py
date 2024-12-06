# Import required packages
import numpy as np
import tensorflow as tf  # For tf.data
import keras
from keras import layers
from keras.applications import EfficientNetB0
import os
import cv2
from matplotlib.pyplot import imread
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt

#Define function to make a model

from tensorflow.keras.applications import EfficientNetB0

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')

    # Freeze the pretrained weights
    model.trainable = False

    # Use inputs in the model processing pipeline
    x = model.output

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    # Build putput layer
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name='EfficientNet')
    optimizer = keras.optimizers.Adam(learning_rate=MODEL1_LR)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'f1_score', 'auc'])

    return model

# Define fucntion to plot accuracy of train and validation dataset from the model

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()



# Define function to unfreeze n layers

def unfreeze_model(model, n=UNFREEZE_LAYERS):
    for layer in model.layers[n:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=MODEL2_LR)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'f1_score', 'auc'])

    return model


# Define function to generate a report of prediction and probabilities
def predict_unseen(model, td, score_threshold, tag='_'):

  folders = [f for f in os.listdir(td) if os.path.isdir(os.path.join(td, f))] # Filter for directories only
  results = [] # Empty list to hold values later

  # For each folder
  for folder in folders:
    td_folder = td+folder+'/'
    file_names = [f for f in os.listdir(td_folder) if os.path.isfile(f'{td_folder}{f}') and f.endswith('.png')]
    # For each png file
    for file in file_names:
      img_path = td_folder+file
      # Read image and process
      img = cv2.imread(img_path)
      img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      x = np.expand_dims(img, axis=0)

      # Use model to predict class
      preds=model.predict(x)
      # Use score threshold to use probabilities to classify
      if preds > score_threshold: ans='yes'
      else: ans='no'

      # Create report
      results.append({
          'file': file,
          'folder': folder,
          'result': ans,
          'probability %': preds*100})

  # Export report as csv
  results_df = pd.DataFrame(results)
  results_df.to_csv(f'{td}Result_report_{tag}.csv', index=False)



