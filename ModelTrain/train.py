from ultralytics import YOLO
from multiprocessing import freeze_support


def train_model():
    # Load a model
    model = YOLO("yolov8n.yaml")

    # Use the model
    results = model.train(data="data.yaml", epochs=3)


if __name__ == '__main__':
    # Freeze support for Windows
    freeze_support()

    train_model()


