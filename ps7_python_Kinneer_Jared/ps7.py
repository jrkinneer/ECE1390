import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from tqdm import tqdm
import os

#1a
car_1 = cv2.imread("./input/p1/car.jpg")
car_hog, car_hog_img = hog(car_1, cells_per_block = (2,2,), channel_axis = 2, visualize = True)
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
            