from flask import Flask, render_template, request
from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(_name_)

# Load model
model = load_model('our_model.h5')
model.make_predict_function()

# Prediction function
def predict_label(img_path):
    try:
        # Try loading and processing the image
        img = image.load_img(img_path, target_size=(224, 224))
        imagee = image.img_to_array(img)
        imagee = np.expand_dims(imagee, axis=0)
        img_data = preprocess_input(imagee)
        prediction = model.predict(img_data)

        # Return prediction results
        if prediction[0][0] > prediction[0][1]:
            return 'Person is safe.'
        else:
            return 'Person is affected by Pneumonia.'
    except Exception as e:
        # If image processing fails, return None
        return None

# Route to main page
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

# Route to handle image submission
@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files.get('my_image')

        # Check if image is uploaded
        if img is None or img.filename == '':
            return render_template("index.html", 
                                   prediction="No image uploaded. Please upload an image.", 
                                   img_path=None)

        # Define the path where the image is supposed to be in the static folder
        img_path = "static/" + img.filename
        
        # Check if the image does not exist in the static folder
        if not os.path.exists(img_path):
            # Show "Invalid image" message and display a placeholder image
            invalid_img_path = "static/invalid_image.jpg"  # Make sure this file exists in the static folder
            return render_template("index.html", 
                                   prediction="Invalid image. Please upload a valid image.", 
                                   img_path=invalid_img_path)

        # If the image exists, proceed with prediction
        p = predict_label(img_path)

        # If prediction is None, show an invalid image and error message
        if p is None:
            invalid_img_path = "static/invalid_image.jpg"
            return render_template("index.html", 
                                   prediction="Invalid image. Please upload a valid image.", 
                                   img_path=invalid_img_path)

        # If prediction is successful, display the result with the uploaded image
        return render_template("index.html", prediction=p, img_path=img_path)

if _name_ == '_main_':
    # app.debug = True
    app.run(debug=True)
