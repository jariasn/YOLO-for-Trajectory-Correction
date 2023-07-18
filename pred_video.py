from ultralytics import YOLO

# Load a model
model = YOLO('runs/segment/train/weights/best.pt')  # load a custom model

video = 'webcam_lane.mp4'

results = model(video)  # predict on an image
model.predict(video, max_det=1, save=True)
