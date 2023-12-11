# app.py
from flask import Flask, render_template
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


# @app.route("/", method=["POST", "GET"])
# def predict_image():
    

if __name__ == '__main__':
    app.run(debug=True, port=8000)