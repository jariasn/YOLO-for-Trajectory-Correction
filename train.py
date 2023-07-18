from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model with Apple Silicon
model.train(data='254473_import-images_001/data_config.yaml', epochs=20, imgsz=640, device='mps', workers=0, pretrained=True, patience=10, lr0=0.001, lrf=0.001, weight_decay=0.0008)
