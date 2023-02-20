from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the Keras model and class labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Define the Flask app
app = Flask(__name__)

# Define the endpoint to accept image uploads and return predictions
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded image file from the request
        file = request.files["image"]

        # Prepare the image for the Keras model
        image = Image.open(file.stream).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make a prediction using the Keras model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Return the prediction result as an HTML template
        if(index==0):
            result = {"class": class_name[2:], "prediction": "Recyclable", "confidence": str(confidence_score)}
        else:
            result = {"class": class_name[2:], "prediction": "Non Recyclable", "confidence": str(confidence_score)}
        return render_template("result.html", result=result)
    else:
        return render_template("index.html")

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
