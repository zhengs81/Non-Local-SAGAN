import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from scipy.linalg import sqrtm

# Load Inception model
model = models.inception_v3(pretrained=True, transform_input=False)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
real_dataset = ImageFolder(root='data/CelebA/img_align_celeba', transform=transform)
generated_dataset = ImageFolder(root='inception/', transform=transform)

real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)
generated_loader = DataLoader(generated_dataset, batch_size=64, shuffle=False)

def get_features(loader):
    features = []
    for batch, _ in loader:
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch)
        features.append(pred.cpu().numpy())
    return np.concatenate(features, axis=0)

# Extract features
real_features = get_features(real_loader)
generated_features = get_features(generated_loader)

# Calculate mean and covariance
mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)

mu_generated = np.mean(generated_features, axis=0)
sigma_generated = np.cov(generated_features, rowvar=False)

# Calculate FID
ssdiff = np.sum((mu_real - mu_generated)**2.0)
covmean = sqrtm(sigma_real.dot(sigma_generated))

if np.iscomplexobj(covmean):
    covmean = covmean.real

fid = ssdiff + np.trace(sigma_real + sigma_generated - 2.0 * covmean)

print(f'FID score: {fid}')
