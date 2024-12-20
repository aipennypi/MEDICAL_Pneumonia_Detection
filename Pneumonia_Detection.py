PATH_PATTERN = './dataset/train/*/*.jpeg'
CLASSES = ['Pneumonia', 'Normal']

import tensorflow as tf
import io
nb_images = len(tf.io.gfile.glob(PATH_PATTERN))
print ('number of images:',nb_images)
print("Pattern matches {} images.".format(nb_images))
filenames_dataset = tf.data.Dataset.list_files(PATH_PATTERN)
for filename in filenames_dataset.take(10):
  print(filename.numpy().decode('utf-8'))

# copy-pasted from "useful code snippets" below
def decode_jpeg(filename):
  bits = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(bits)
  return image

image_dataset = filenames_dataset.map(decode_jpeg)
for image in image_dataset.take(10):
  print(image.numpy().shape)


# copy-pasted from "useful code snippets" below
def decode_jpeg_and_label(filename):
  bits = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(bits)
  label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
  label = label.values[-2]
  return image, label

dataset = filenames_dataset.map(decode_jpeg_and_label)
for image, label in dataset.take(8):
  print(image.numpy().shape, label.numpy().decode('utf-8'))

import tensorflow as tf
import io

def crop_center_and_resize(img, size): #224*224
    s = tf.shape(img)
    w, h = s[0], s[1]
    c = tf.minimum(w, h)
    w_start = (w - c) // 2
    h_start = (h - c) // 2
    center = img[w_start:w_start + c, h_start:h_start + c]
    return tf.image.resize(img, [size, size])

#for i, (image, label) in enumerate(dataset):
#  crop_image = crop_center_and_resize(image, 224)

def normalize_image(image, mean, std):
    for channel in range(3):
        image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
    return image
img = []
label = []
for filename in filenames_dataset.take(800):#nb_images
  print(filename.numpy().decode('utf-8'))
  image, label_ = decode_jpeg_and_label(filename) # assign return values to temporary variables
  #image = normalize_image(np.array(image) / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  image = crop_center_and_resize(image, 224) # crop image
  img.append(image) # append the image to the listaxa
  label.append(label_) # append the label to the list
  print(img[-1].numpy().shape, label[-1].numpy().decode('utf-8')) # access last element using -1

from tensorflow.keras.preprocessing.image import ImageDataGenerator #Import the ImageDataGenerator class

# Here we are going to use tensorflow "ImageDataGenerator" module.
train_data_folder = './Pneumonia/dataset/train'
valid_data_folder = './Pneumonia/dataset/valid'
test_data_folder = './Pneumonia/dataset/test'

img_size = (224, 224)
batch_size = 60

print("Getting Data...")
datagen = ImageDataGenerator(rescale=1./255, # normalize pixel values
                             validation_split=0.3) # hold back 30% of the images for validation

print("Preparing training dataset...")
train_generator = datagen.flow_from_directory(
    train_data_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

print("Preparing validation dataset...")
validation_generator = datagen.flow_from_directory(
    valid_data_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

print("Preparing test dataset...")
test_generator = datagen.flow_from_directory(
    test_data_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as test data

classnames = list(train_generator.class_indices.keys())
print('Data generators ready')

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=0)
images, labels = next(img_gen.flow_from_directory(train_data_folder))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.patches as patches
import os
import cv2
import torch
from PIL import Image

complete_images = []
complete_class = []

#earlt stop
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = "val_acc",
                    mode='max',
                    min_delta = 0.0001,
                    verbose=1,
                    patience = 5,
                    restore_best_weights = True,
                    baseline = None)

#CNN

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Dense
from tensorflow.keras.optimizers import Adam

model = models.Sequential()

# Create a new model by adding your classification layers
model.add(keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'),)
model.add(keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),)#model.add(layers.BatchNormalization()),
model.add(keras.layers.MaxPooling2D(),)
# Second Convolutional Layer
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),)
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),)
#model.add(layers.BatchNormalization()),
model.add(keras.layers.MaxPooling2D(),)

# Third Convolutional Layer
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),)
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),)
#model.add(layers.BatchNormalization()),
model.add(keras.layers.MaxPooling2D(),)
# 4th Convolutional Layer
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),)
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),)
#model.add(layers.BatchNormalization()),
model.add(layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D(),)
# 5th Convolutional Layer
model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),)
model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),)
#model.add(layers.BatchNormalization()),
model.add(layers.Dropout(0.2))
model.add(keras.layers.MaxPooling2D(),)

# sixth Convolutional Layer
#model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),)
model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),)
model.add(keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),)
model.add(layers.BatchNormalization()),
model.add(layers.Dropout(0.35))
model.add(keras.layers.MaxPooling2D(),)'
model.summary()

#TRAIN THE MODEL
import numpy as np

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming 'Pneumonia' is one class and 'Normal' is the other
label_mapping = {'Pneumonia': 0, 'Normal': 1} # Create a dictionary to map labels to integers

X_train = np.array([np.resize(img.numpy(), (224, 224, 3)) for img in df['Images'].tolist()])
y_train = np.array([label_mapping[label.numpy().decode('utf-8')] for label in df['Class']]) # Decode byte strings and map to integers

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
cnn_train_history = model.fit(X_train, y_train, epochs=24, batch_size=32, validation_split=0.2,callbacks=[es])
# VGG

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dropout, LayerNormalization, Dense
from tensorflow.keras.optimizers import Adam
from keras import applications
from keras.applications.VGG16  import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = True
model = models.Sequential()
model.add(base_model)

# Flatten layer to transition from convolutional to dense layers
model.add(keras.layers.Flatten(),)

# Dense (Fully Connected) Layers
model.add(keras.layers.Dense(128, activation='relu'),)
model.add(layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))  # Binary classification, so use 1 output with sigmoid activation
model.summary()
#TRAIN THE MODEL
import numpy as np

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming 'Pneumonia' is one class and 'Normal' is the other
label_mapping = {'Pneumonia': 0, 'Normal': 1} # Create a dictionary to map labels to integers

X_train = np.array([np.resize(img.numpy(), (224, 224, 3)) for img in df['Images'].tolist()])
y_train = np.array([label_mapping[label.numpy().decode('utf-8')] for label in df['Class']]) # Decode byte strings and map to integers

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
vgg_train_history = model.fit(X_train, y_train, epochs=24, batch_size=32, validation_split=0.2,callbacks=[es])
#draw the metrics
import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()

def plot_images_labels_predict(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "l=" + str(labels[idx])
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))
            print("\n")
        else:
            title = "l={}".format(str(labels[idx]))
            print("\n")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx+=1
    plt.show()

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#Evaluation
T_PATH = './dataset/test/*/*.jpeg'
tnb_images = len(tf.io.gfile.glob(T_PATH))
print ('number of images:',nb_images)
print("Pattern matches {} images.".format(nb_images))
filenames_dataset = tf.data.Dataset.list_files(T_PATH)
for filename in filenames_dataset.take(nb_images):
  print(filename.numpy().decode('utf-8'))
t_img = []
t_label = []
for filename in filenames_dataset.take(tnb_images):
  print(filename.numpy().decode('utf-8'))
  image, label_ = decode_jpeg_and_label(filename) # assign return values to temporary variables
  #image = normalize_image(np.array(image) / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  image = crop_center_and_resize(image, 224) # crop image
  t_img.append(image) # append the image to the listaxa
  t_label.append(label_) # append the label to the list
  print(img[-1].numpy().shape, label[-1].numpy().decode('utf-8')) # access last element using -1

t_complete_images = []
t_complete_class = []
t_complete_images = t_img
t_complete_class = t_label
df_t = pd.DataFrame()
df_t['Images'] = t_complete_images
df_t['Class'] = t_complete_class
#df['Images']/=255
df_t['Images'].shape
import numpy as np
label_mapping = {'Pneumonia': 0, 'Normal': 1}
X_test = np.array([np.resize(img.numpy(), (224, 224, 3)) for img in df_t['Images'].tolist()])
y_test = np.array([label_mapping[label.numpy().decode('utf-8')] for label in df_t['Class']]) # Decode byte strings and map to integers
print("X_test:", X_test.shape, X_test.dtype)
print("y_test:", y_test.shape, y_test.dtype)
model.evaluate(X_test,y_test)
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

y_pred = model.predict(X_test, verbose=0)
if isinstance(y_pred, tf.RaggedTensor):
  y_pred = y_pred.to_tensor()
y_pred_classes = np.argmax(y_pred, axis=1)
label_names = ['Fall', 'No Fall', 'Sitting']
#print (y_pred_classes)
#print (y_test)


# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix' )
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.savefig(f'confusion_matrix_{model_name}.png')
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

y_pred = model.predict(X_test, verbose=0)
if isinstance(y_pred, tf.RaggedTensor):
  y_pred = y_pred.to_tensor()
y_pred_classes = np.argmax(y_pred, axis=1)
label_names = ['Fall', 'No Fall', 'Sitting']
#print (y_pred_classes)
#print (y_test)


# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix' )
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.savefig(f'confusion_matrix_{model_name}.png')
plt.show()
