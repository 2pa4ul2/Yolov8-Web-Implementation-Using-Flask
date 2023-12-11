# predict.py
import torch
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, plot_one_box
from utils.torch_utils import select_device

def predict_image(image_path):
    device = select_device('')  # Selects the first available GPU or CPU if no GPU is available
    model = attempt_load('best.pt', map_location=device)  # Load the YOLOv8 model
    img0 = cv2.imread(image_path)  # BGR

    # Preprocess the image
    img = letterbox(img0, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # to BGR
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run inference
    pred = model(img)[0]

    # Post-process the predictions
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / img0.shape).view(-1).tolist()
                label = f'Class {int(cls)}: {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=3)

    result_path = 'static/results/' + Path(image_path).name
    cv2.imwrite(result_path, img0)

    return result_path
