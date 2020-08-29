from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model

with open('model_config.json') as file:
  json_config = file.read()
model = tf.keras.models.model_from_json(json_config)
model.load_weights('model')


def model_predict(image_path, model):
  
  test_image = image.load_img(image_path, target_size = (264, 264))
  test_image = image.img_to_array(test_image)
  test_image = preprocess_input(test_image)
  test_image = np.expand_dims(test_image, axis=0)

  result = model.predict(test_image)
  print(result)
  result = result.argmax(axis = 1)[0]

  if result==0:
    prediction = 'Its a Diseased Cotton Leaf'
  elif(result == 1):
    prediction = 'Its a Diseased Cotton Plant'
  elif(result == 2):
    prediction = 'Its a Fresh Cotton Leaf'
  else:
    prediction = 'Its a Fresh Cotton Plant'
  
  return prediction


@app.route('/', methods=['GET'])
def index():

  return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
  
  if request.method == 'POST':
    file = request.files['file']
    file_path = './'+file.filename
    file.save(file_path)

    prediction = model_predict(file_path, model)
    return prediction
 
  return None


if __name__ == '__main__':
    app.run()
