from flask import Flask, render_template, request, redirect, url_for,send_from_directory
import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from PIL import Image

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
    opt.conf = 0.5
    results = model(**vars(opt), stream=True)

    for i, result in enumerate(results):
        if opt.save_txt:
            result_json = json.loads(result.tojson())
            yield json.dumps({'results': result_json})
        else:
            im0 = result.plot()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            im_path = save_path / f"result_image_{timestamp}_{i}.jpg"
            cv2.imwrite(str(im_path), im0)
            im_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im_bytes + b'\r\n')

# Splash page
@app.route('/')
def splash():
    return render_template('base.html')

#gallery page
@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    return render_template('gallery.html')

# @app.route('/gallery', methods=['GET', 'POST'])
# def gallery():
#     pics = os.path.join(app.config["UPLOAD_FOLDER"], 'result_image.jpg')

#     imagelist = os.listdir('results')
#     imagelist = ['results/' + image for image in imagelist]
#     return render_template('gallery.html', imagelist=imagelist)


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


#Detection page
@app.route('/detection', methods=['GET', 'POST'])
def detection():
    saved_filenames = request.args.getlist('filenames')
    result_path = Path(__file__).parent / 'static' / 'results'
    list_of_images = sorted(result_path.glob('result_image_*.jpg'), key=os.path.getmtime, reverse=True)

    if list_of_images:
        most_recent_image = list_of_images[0]
    else:
        most_recent_image = None

    return render_template('detection.html', most_recent_image=most_recent_image, saved_filenames=saved_filenames)
    
    
# Index page
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')
        save_txt = request.form.get('save_txt', 'F')

        if uploaded_file:
            source = Path(__file__).parent / raw_data / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
        else:
            opt.source, _ = update_options(request)

        opt.save_txt = True if save_txt == 'T' else False

        result_path = Path(__file__).parent / 'static' / 'results'  
        result_path.mkdir(parents=True, exist_ok=True)
        predictions = list(predict(opt, save_path=result_path))

        saved_filenames = [f"result_image_{i}.jpg" for i in range(len(predictions))]

        return redirect(url_for('detection', filenames=saved_filenames))
    return render_template('index.html')




if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--weights', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')
    parser.add_argument('--conf','--conf-thres', type=float, default=0.7, help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--show','--view-img', default=False, action='store_true', help='show results if possible')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--save_txt','--save-txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels','--show-labels', default=True, action='store_true', help='show labels')
    parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
    parser.add_argument('--max_det','--max-det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true', help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--line_width', '--line-thickness', default=None, type=int, help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
    parser.add_argument('--augment', default=False, action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--classes', type=list, help='filter results by class, i.e. classes=0, or classes=[0,2,3]') # 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
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
    