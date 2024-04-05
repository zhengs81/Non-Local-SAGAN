import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

# Function to preprocess images
def preprocess_images(images):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    return torch.stack([transform(image) for image in images])

# Function to calculate Inception Score
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i in range(0, N, batch_size):
        batch = imgs[i:i + batch_size]
        batch = batch.type(dtype)
        batchv = torch.autograd.Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i:i + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"): # Adjust the extension based on your image format
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    images.append(img.convert('RGB'))
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return images

folder_path = 'inception/sagan_celeb' # Adjust based on your folder path
images = load_images_from_folder(folder_path)
print(len(images))

preprocessed_imgs = preprocess_images(images)

# Calculate the Inception Score
is_mean, is_std = inception_score(preprocessed_imgs, cuda=True, batch_size=32, resize=True, splits=10)

print(f"Inception Score: Mean = {is_mean}, Std = {is_std}")
