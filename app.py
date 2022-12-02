import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from keras.models import load_model

app = Flask(__name__)

model = load_model('model.h5')

@app.route("/",methods = ["POST"])
def predict():
    text = request.form.get('inputText')
    input_query = np.array([text])
    result = model.predict(input_query)[0]
    return jsonify({"Prediction":str(result)})


if __name__ == "__main__":
    app.run(debug=True)


