from flask import Flask, render_template, request
from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import pickle
import numpy as np


app = Flask(__name__)

model = load_model('our_model.h5')

model.make_predict_function()

def predict_label(img_path):
    img=image.load_img(img_path,target_size=(224,224))
    imagee=image.img_to_array(img)
    imagee=np.expand_dims(imagee, axis=0)
    img_data=preprocess_input(imagee)
    prediction=model.predict(img_data)
    if prediction[0][0]>prediction[0][1]:
        return 'Person is safe.'
    else:
        return 'Person is affected with Pneumonia.'



@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
	    