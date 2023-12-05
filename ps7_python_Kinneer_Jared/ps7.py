import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from tqdm import tqdm
import os

#1a
car_1 = cv2.imread("./input/p1/car.jpg")
car_hog, car_hog_img = hog(car_1, cells_per_block = (2,2,), channel_axis = 2, visualize = True)
print(car_hog)
# cv2.imshow("hog", car_hog_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite("./output/ps7-1-a.png", car_hog_img)

#1b
main_path = "./input/p1/train_imgs"
X_train = np.zeros((1, car_hog.shape[0]))
y_train = np.zeros((1,1))

for file in tqdm(os.listdir(main_path), ):
    img = cv2.imread(main_path + "/" + file)
    hog_vector = hog(img, cells_per_block = (2,2,), channel_axis = 2)
    np.vstack((X_train, hog_vector))
    
    label = int(file.split("_")[0])
    np.vstack((y_train, label))
    
#remove placeholder row from X_train and y_train
X_train = X_train[1:]
y_train = y_train[1:]
#report
print(X_train.shape + "\n" + y_train.shape)

#1c
svm_classifier = LinearSVC()
svm_classifier.fit(X_train, y_train)

test_path = "./input/p1/test_imgs"

window_size = (96, 32)
for file in os.listdir(test_path):
    img = np.asarray(cv2.imread(test_path + "/" + file))
    for i in range(img.shape[0] - window_size[0]):
        for j in range(img.shape[1] - window_size[1]):
            window = img[i:window_size[0], j:window_size[1]]
            
            window_hog = hog(window, cells_per_block = (2,2,), channel_axis = 2)
            
            results_vector = svm_classifier.decision_function(window_hog)
    hog_vector = hog(img, cells_per_block = (2,2,), channel_axis = 2)
    
    