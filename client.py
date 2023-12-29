import requests
# Saving txt file
resp = requests.get('http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg&save_txt=T',
                    verify=False)
print(resp.content)

# Without save txt file, just labeling the image
resp = requests.get('http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg',
                    verify=False)
print(resp.content)

# You can also copy and paste the following url in your browser
'http://0.0.0.0:5000/predict?source=https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg'


resp = requests.get('http://0.0.0.0:5000/predict?source=https://www.youtube.com/watch?v=MNn9qKG2UFI',
                    verify=False)

'http://0.0.0.0:5000/predict?source=https://www.youtube.com/watch?v=MNn9qKG2UFI'



url = 'http://0.0.0.0:5000/predict'
file_path = 'data/images/bus.jpg'

params = {
    'save_txt': 'T'
}

with open(file_path, "rb") as f:
    response = requests.post(url, files={"myfile": f}, data=params, verify=False)

print(response.content)