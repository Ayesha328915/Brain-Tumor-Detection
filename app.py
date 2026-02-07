from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)

# Load trained model
model = load_model('brain_tumor_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels = ['No Tumor', 'Tumor']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    img_path = ""
    confidence = None

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            # Image preprocessing
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediction
            pred = model.predict(img_array)
            confidence = float(np.max(pred)) * 100
            result = np.argmax(pred)

            prediction = labels[result]

    return render_template(
    'index.html',
    prediction=prediction,
    confidence=confidence,
    img_path=img_path
)


if __name__ == '__main__':
    app.run(debug=True)
