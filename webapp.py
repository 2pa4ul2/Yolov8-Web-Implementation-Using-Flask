from flask import Flask, render_template, request, redirect, url_for,send_from_directory
import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils.checks import cv2, print_args
from utils.general import update_options

# Initialize paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Initialize Flask API
app = Flask(__name__)
app.config['IMAGE_RESULTS'] = "static/results"

def predict(opt, save_path=None):
    for i, result in enumerate(model(**vars(opt), stream=True)):
        labels_for_image = []

        # Sort bounding boxes by their x-coordinate to maintain order
        sorted_indices = sorted(range(len(result.boxes.xyxy)), key=lambda k: result.boxes.xyxy[k][0])

        for idx in sorted_indices:
            c = result.boxes.cls[idx]
            label = model.names[int(c)]
            confidence = result.boxes.conf[idx]
            labels_for_image.append(f"{label} ({confidence:.2f})")
            print(f"{label} ({confidence:.2f})")

        im0 = result.plot()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        im_path = save_path / f"result_image_{timestamp}_{i}.jpg"
        cv2.imwrite(str(im_path), im0)
        im_bytes = cv2.imencode('.jpg', im0)[1].tobytes()

        yield im_bytes, labels_for_image

# Splash page
@app.route('/')
def splash():
    return render_template('base.html')

#gallery page
@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    # Get the path to the "results" folder
    result_path = Path(__file__).parent / 'static' / 'results'

    # Get the list of image filenames in the "results" folder
    image_filenames = sorted(result_path.glob('result_image_*.jpg'), key=os.path.getmtime, reverse=True)

    # Pass the list of image filenames to the template
    return render_template('gallery.html', image_filenames=image_filenames)



@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')
        save_txt = request.form.get('save_txt', 'F')

        if uploaded_file:
            source = Path(__file__).parent / 'raw_data' / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
        else:
            opt.source, _ = update_options(request)

        opt.save_txt = True if save_txt == 'T' else False

        result_path = Path(__file__).parent / 'static' / 'results'  
        result_path.mkdir(parents=True, exist_ok=True)
        # Call the modified predict function to get the image bytes and labels
        im_bytes, labels_list = zip(*predict(opt, save_path=result_path))
        saved_filenames = [f"result_image_{i}.jpg" for i in range(len(im_bytes))]

        return render_template('detection.html', most_recent_image=None, saved_filenames=saved_filenames, im_bytes=im_bytes, labels_list=labels_list)

    return render_template('index.html')


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--weights', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')
    parser.add_argument('--conf','--conf-thres', type=float, default=0.33, help='object confidence threshold for detection')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

    # print used arguments
    print_args(vars(opt))

    # Get por to deploy
    port = opt.port
    delattr(opt, 'port')
    
    # Create path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')
    
    # Load model
    model = YOLO(str(opt.model))

    # Run app
    app.run(host='0.0.0.0', port=port, debug=False)
    