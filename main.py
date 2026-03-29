# main.py
from train import train_model
from inference import predict

if __name__ == "__main__":
    print("Training model...")
    train_model()

    print("Testing inference:")
    print(predict("I have nausea and left chest pain and dizziness"))
