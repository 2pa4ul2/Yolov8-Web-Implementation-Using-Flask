from flask import Flask, render_template, request, redirect, url_for,send_from_directory  #Flask Dependency
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
app.config['IMAGE_RESULTS'] = "static/results" #Prediction Results Path

#Prediction
def predict(opt, save_path=None):
    # opt.conf = 0.5
    results = model(**vars(opt), stream=True) #Make Predictions

    for i, result in enumerate(results):
        if opt.save_txt:
            result_json = json.loads(result.tojson())       #Convert to json
            yield json.dumps({'results': result_json})      #yield json representation
        else:
            im0 = result.plot()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")           #Timestamp for filename
            im_path = save_path / f"result_image_{timestamp}_{i}.jpg"       #Path,Filename,index
            cv2.imwrite(str(im_path), im0)                                  #save to path
            im_bytes = cv2.imencode('.jpg', im0)[1].tobytes()               
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + im_bytes + b'\r\n')

# Splash page
@app.route('/')
def splash():
    return render_template('base.html') #Render Base.html

#Gallery page
@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    result_path = Path(__file__).parent / 'static' / 'results'                                            # Path to the "results" folder
    image_filenames = sorted(result_path.glob('result_image_*.jpg'), key=os.path.getmtime, reverse=True)  # Get the list of image filenames in the "results" folder
    
    return render_template('gallery.html', image_filenames=image_filenames)                               # Pass the list of image filenames to the template



@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html') #Render about.html


#Detection page
@app.route('/detection', methods=['GET', 'POST'])
def detection():
    saved_filenames = request.args.getlist('filenames')         #Retrieve list of saved filenames
    result_path = Path(__file__).parent / 'static' / 'results'  #Path of predicted images
    list_of_images = sorted(result_path.glob('result_image_*.jpg'), key=os.path.getmtime, reverse=True) #List all and Sort by time

    if list_of_images:
        most_recent_image = list_of_images[0]   #if images exist, set the most recent as first image
    else:
        most_recent_image = None        #if no images, set none

    return render_template('detection.html', most_recent_image=most_recent_image, saved_filenames=saved_filenames)
    
    
# Index page
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile')                                     #Get uploaded image
        save_txt = request.form.get('save_txt', 'F')

        if uploaded_file:                                                               #If the File exist save to specifict path
            source = Path(__file__).parent / raw_data / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source
        else:
            opt.source, _ = update_options(request)

        opt.save_txt = True if save_txt == 'T' else False

        result_path = Path(__file__).parent / 'static' / 'results'                      #Directory to save the Predictions
        result_path.mkdir(parents=True, exist_ok=True)                                  #Create directory if it does not exist
        predictions = list(predict(opt, save_path=result_path))                         #Generate Prediction and save it to path

        saved_filenames = [f"result_image_{i}.jpg" for i in range(len(predictions))]    #Renaming image with the new name for uniformity

        return redirect(url_for('detection', filenames=saved_filenames))                #Return detection page to display
    return render_template('index.html')




if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--weights', type=str, default=ROOT / 'best.pt', help='model path or triton URL')     #Necessary for loading the model
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')   
    parser.add_argument('--conf','--conf-thres', type=float, default=0.33, help='object confidence threshold for detection')    #Confidence level: currently set as 33%
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')  
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')  #Saving Images with no bounding box  
    parser.add_argument('--port', default=5000, type=int, help='port deployment')   #PORT
    opt, unknown = parser.parse_known_args()

    # print used arguments
    print_args(vars(opt))

    # Get port to deploy
    port = opt.port
    delattr(opt, 'port')
    
    # Creating path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')
    
    # Load Yolov8 model
    model = YOLO(str(opt.model))

    app.run(host='0.0.0.0', port=port, debug=False)
    