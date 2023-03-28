from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)

scaler = joblib.load('/home/mohammad/Documents/classification/scaler.save')
le = joblib.load('/home/mohammad/Documents/classification/labelEncoder.save')
width, height = 224, 224
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_content = file.read()
        img = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (width, height))
        img = np.reshape(img, (1, -1))
        img = scaler.transform(img)
        img = np.reshape(img, (1, width, height, 3))
        prediction = model.predict(img)
        prediction = le.inverse_transform(np.argmax(prediction, axis=1))[0]
        image_url = "data:image/jpeg;base64," + base64.b64encode(file_content).decode("utf-8")
        return render_template('index.html', prediction=prediction, image_url=image_url)



if __name__ == '__main__':
    app.run(debug=True)
