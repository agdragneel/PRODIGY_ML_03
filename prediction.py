from sklearn import svm
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
import cv2
import os
import pandas as pd

def extract_features(image_path):
    image=cv2.imread(image_path)
    resized_image=cv2.resize(image,(100,100))
    grayscale_image=cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
    flattened_image=grayscale_image.flatten()
    return flattened_image
# Load the trained model
svm_model = joblib.load('svm_model.pkl')
predictions=[]
# Extract features from the test image
file_list = os.listdir("test1\\test1")
for file in tqdm(file_list):
    image_path = os.path.join("test1\\test1", file)
    test_features=extract_features(image_path)
    test_prediction = svm_model.predict([test_features])
    prediction=""
    if test_prediction == 0:
        prediction="Cat"
    else:
        prediction="Dog"

    predictions.append((file,prediction))

df = pd.DataFrame(predictions, columns=['Image_Number', 'Prediction'])
df.to_csv("Predictions.csv",index=False)


