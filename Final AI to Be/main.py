from preprocessing import normalize_image
from inference import load_model
from postprocessing import predict

# * Define image
path_image = "mnist_41_sneaker.png"


if __name__ == "__main__":
    # * PREPROCESSING
    result_image = normalize_image(path_image)

    # * INFERENCE
    model = load_model()

    # * POSTPROCESSING
    predict(model, result_image)
