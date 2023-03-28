# %%
import os
import glob
import cv2
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

root_dir = '/home/mohammad/Documents/classification/cars classification/dataset/DATA/prediction/'
width, height = 224, 224
scaler = joblib.load('/home/mohammad/Documents/classification/scaler.save')
le = joblib.load('/home/mohammad/Documents/classification/labelEncoder.save')

# Resize the images
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        images = glob.glob(os.path.join(subdir_path, '*.jpg'))
        for image in images:
            img = cv2.imread(image)
            img_resized = cv2.resize(img, (width, height))
            cv2.imwrite(image, img_resized)

# Load and preprocess the images
X = [cv2.imread(image) for image in glob.glob(root_dir + '*')]
X = np.array(X)
X = np.reshape(X, (X.shape[0], -1))
X = scaler.transform(X)
X = np.reshape(X, (X.shape[0], width, height, 3))

# Load the model and make predictions
model = load_model('model.h5')
y_original = le.inverse_transform(np.argmax(model.predict(X), axis=1))

# Count the number of occurrences of each prediction
counts = pd.DataFrame(y_original).value_counts()
print(f"\n\nthe predicted cars are \n\n{counts}")






# %%
