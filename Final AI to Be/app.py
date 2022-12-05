import os
import numpy as np
import torch
import numpy as numpy

from PIL import Image
from flask import Flask, flash, request, make_response, render_template, jsonify, request
from flask_restful import Resource, Api
from inference import load_model
from preprocessing import normalize_image
from postprocessing import predict
from initialization import FashionCNN

UPLOAD_FOLDER = 'uploads'
PATH = "best_modell.pt"
app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Your Route


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        name = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

        image_file = Image.open(
            "uploads/{}".format(name))  # open colour image
        # convert image to black and white
        image_file = image_file.convert('1')
        image_file.save('uploads/{}'.format(name))

        with open('uploads/{}'.format(name), 'r+b') as f:
            with Image.open(f) as img:
                path = Image.open('uploads/{}'.format(name)).convert(mode="L")
                img = path.resize((28, 28))

        #! load model
        model = FashionCNN()
        model.load_state_dict(torch.load(PATH))

        # Prepare image for model
        img = numpy.array(img)

        img = numpy.expand_dims(img, axis=0)
        g = (img / 255.0).astype(numpy.float32)
        g = numpy.expand_dims(g, axis=0)
        y = torch.from_numpy(g)

        # Push image to model
        y_pred = model(y)

        # Get prediction
        y_pred = torch.argmax(y_pred, dim=1)

        a = y_pred.detach().numpy()[0]

        # labels of images
        Class_names = ['T-shirt/top',
                       'Trouser',
                       'Pullover',
                       'Dress',
                       'Coat',
                       'Sandal',
                       'Shirt',
                       'Sneaker',
                       'Bag',
                       'Ankle boot']

        # return f"{a}"
        return f"Predicted number is - {Class_names[a]}"
    else:
        return render_template("Index.html")
        # return "test - 2"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


# UPLOAD_FOLDER = 'uploads'
# app = Flask(__name__, template_folder='.')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # Your Route


# @app.route("/", methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['image']
#         name = file.filename
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

#         image_file = Image.open(
#             "uploads/{}".format(name))  # open colour image
#         # convert image to black and white
#         image_file = image_file.convert('1')
#         image_file.save('uploads/{}'.format(name))

#         with open('uploads/{}'.format(name), 'r+b') as f:
#             with Image.open(f) as img:
#                 img_dir = Image.open('uploads/{}'.format(name))
#                 img = img_dir.resize((28, 28))

#                 # * PREPROCESSING
#                 result_image = normalize_image(img_dir)

#         # * INFERENCE
#         model = load_model()

#         # * POSTPROCESSING
#         result = predict(model, result_image)

#         return f"Predicted number is - {result}"
#     else:
#         return render_template("Index.html")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
