import os
from ultralytics import YOLO

# Load a model
model = YOLO('runs/segment/train/weights/best.pt') 

# Directory containing the images
directory = 'datasets/images/test'

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Filter image files
        filepath = os.path.join(directory, filename)

        # Predict with the model
        results = model(filepath)  # predict on an image
        model.predict(filepath, max_det=1, boxes=False, save=True)
