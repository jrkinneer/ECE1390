import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from tqdm import tqdm
import os

#1a
car_1 = cv2.imread("./input/p1/car.jpg")
car_hog, car_hog_img = hog(car_1, cells_per_block = (2,2,), channel_axis = 2, visualize = True)
print(car_hog.shape)
# cv2.imshow("hog", car_hog_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite("./output/ps7-1-a.png", car_hog_img)

#1b
main_path = "./input/p1/train_imgs"
X_train = np.zeros((1, car_hog.shape[0]))
y_train = []

for file in tqdm(os.listdir(main_path), "files in input directory"):
    img = np.asarray(cv2.imread(main_path + "/" + file))
    hog_vector = hog(img, cells_per_block = (2,2,), channel_axis = 2)
    X_train = np.vstack((X_train, hog_vector))
    
    label = int(file.split("_")[0])
    y_train.append(label)
    
#remove placeholder row from X_train and y_train
X_train = X_train[1:]
#report
print(X_train.shape)
print(len(y_train))

#1c
svm_classifier = LinearSVC()
svm_classifier.fit(X_train, y_train)

test_path = "./input/p1/test_imgs"

window_size = (96, 32)
K = 1
#for file in test directory
for file in tqdm(os.listdir(test_path), "file in test directory", leave=False):
    img = np.asarray(cv2.imread(test_path + "/" + file))
    
    i_j_list = []
    car_prob = []
    
    #for all possible window locations
    for i in tqdm(range(img.shape[0] - window_size[0]), "i for classifier window", leave=False):
        for j in tqdm(range(img.shape[1] - window_size[1]), "j for classifier window", leave=False):
            #get window
            window = img[i:i+window_size[0], j:j+window_size[1]]
            
            #window hog
            window_hog = hog(window, cells_per_block = (2,2,), channel_axis = 2)
            window_hog = np.reshape(window_hog, (1,-1))
            #predict car or not car
            results_vector = svm_classifier.decision_function(window_hog)
            
            #record data
            i_j_list.append((i,j))
            car_prob.append(results_vector[0])
            
    #get highest prob in car_prob
    ind = np.argmax(car_prob)
    val = np.max(car_prob)
        
    #draw rectangle if above threshold and save image regardless
    #orignal threshold of 1.2 couldn't identify any car, and gave one false positive
    if val > .9:
        start_point = (i_j_list[ind][1], i_j_list[ind][0])
        end_point = (start_point[0] + window_size[1], start_point[1] + window_size[0])
        rectangle_img = cv2.rectangle(img, start_point, end_point, color=(0,255,0))
        cv2.imwrite("./output/ps7-1-d-"+str(K)+".png", rectangle_img)
    else:
        cv2.imwrite("./output/ps7-1-d-"+str(K)+".png", img)
        
    K+=1
  
#2       

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow import keras
# from tensorflow.keras import layers
# Load training and validation sets
ds_train_ = tf.keras.preprocessing.image_dataset_from_directory(
    './input/p2/train_imgs',
    labels='inferred',
    label_mode='categorical',
    image_size=[32, 32],
    batch_size=100,
    shuffle=True,
)
ds_test_ = tf.keras.preprocessing.image_dataset_from_directory(
    './input/p2/test_imgs',
    labels='inferred',
    label_mode='categorical',
    image_size=[32, 32],
    batch_size=100,
    shuffle=False,
)
# Define the model
model = tf.keras.Sequential([
    # First Convolutional Block
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
    # give the input dimensions in the first layer
    # [height, width, color channels(RGB)]
    input_shape=[32, 32, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    # Second Convolutional Block
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    # third Convolutional Block
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation="relu", padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(),
    # Classifier Head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=27, activation="relu"),
    tf.keras.layers.Dense(units=3, activation="softmax"),
    
])
model.summary()
#train the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    ds_train_,
    #validation_data=ds_valid_,
    epochs=10,
)
plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy') # validation accuracy; no validation in this example
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
#test the model
scores = model.evaluate(ds_test_, verbose=0)
print('Accuracy on testing data: {}% \n Error on training data: {} \n'.format(scores[1], 1 - scores[1]))
print(model.predict(ds_test_))
# #displaying an image with the detected label
# from keras.preprocessing import image
# im = image.load_img('./input/p2/test_imgs/Blue/2_00.jpg')
# im = np.expand_dims(im, axis=0)
# pred = model.predict(im, verbose=0)
# print(pred)
# classes = ["Black", "Blue", "Green", "No car"]
# class_ID = np.argmax(pred)
# title = 'predicted ' + classes[class_ID]
# plt.imshow(tf.squeeze(im))
# plt.axis('off')
# plt.title(title)
# plt.show()
