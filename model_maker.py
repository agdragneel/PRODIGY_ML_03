from sklearn import svm
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
import cv2
import os

def extract_features(image_path):
    image=cv2.imread(image_path)
    resized_image=cv2.resize(image,(100,100))
    grayscale_image=cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    flattened_image=grayscale_image.flatten()
    return flattened_image

train_data=[]
train_labels=[]

print("Reading Images")

file_list = os.listdir("train\\train")
np.random.shuffle(file_list)
print(file_list)
data_size=10000
count = 0
for file in tqdm(file_list):
    if count >= data_size:
        break
    if file.startswith("cat"):
        label = 0
    elif file.startswith("dog"):
        label = 1
    else:
        continue
    count=count+1

    image_path = os.path.join("train\\train", file)
    features = extract_features(image_path)
    train_data.append(features)
    train_labels.append(label)



svm_model=svm.SVC(kernel='linear')
print("Fitting Model")
svm_model.fit(train_data,train_labels)
print("Fitting Done")
joblib.dump(svm_model, 'svm_model.pkl')
