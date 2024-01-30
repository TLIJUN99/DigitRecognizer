import os
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor

"""Load MNIST Pretrained ViT Model """
pretrained_model = os.environ['MNIST_MODEL_PATH']
processor = ViTImageProcessor.from_pretrained(pretrained_model)
model = AutoModelForImageClassification.from_pretrained(pretrained_model)

# pretrained_model = "farleyknight-org-username/vit-base-mnist"
# processor = ViTImageProcessor.from_pretrained(pretrained_model)
# model = AutoModelForImageClassification.from_pretrained(pretrained_model)



def predict_digit(img):
    """
    Predict a digit image.

    Args:
        img (PIL.Image.Image): The input image to be classified.

    Returns:
        A numpy array of probabilities for each digit class (0-9).
    """
    # Preprocess the input image using the image processor
    inputs = processor(images=img, return_tensors="pt")

    # Pass the preprocessed inputs through the model
    prob = model(**inputs)

    # Softmax the model output to get probabilities for each class
    prob = torch.nn.functional.softmax(prob.logits, dim=1)[0]

    # Detach from the computational graph and convert to numpy array
    prob = prob.detach().numpy()

    return prob
