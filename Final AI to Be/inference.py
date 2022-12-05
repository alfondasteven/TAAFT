import torch
import numpy
from PIL import Image
from initialization import FashionCNN


def load_model():
    # Load model
    PATH = "best_modell.pt"
    model = FashionCNN()
    model.load_state_dict(torch.load(PATH))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return model
