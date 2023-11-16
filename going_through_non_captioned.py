# Get all image paths from non captioned
# Loop through them with tqdm
# Open the images with pil, pass them through the model
# Convert logits into pred_probs
# Store pred_prob: image_path
# Sort the dictionary and get the three best results

from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch, torchvision
from torch import nn

images_path = Path("non-captioned/images")
paths = list(images_path.glob('*.jpg'))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet50()
model.fc = nn.Linear(2048, 1)
model.conv1 = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=1, stride=1, padding=0, bias=False),
    nn.Conv2d(10, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
)
model.load_state_dict(torch.load("models/resnet_mars_landing_site_predictor.pth"))

transform = ResNet50_Weights.DEFAULT.transforms()
model.to(device)

data_dict = {}

for path in tqdm(paths):
    img = Image.open(path).convert('RGB')
    pred = torch.sigmoid(model(transform(img).unsqueeze(0).to(device))).item()
    data_dict[pred] = path

sorted_dict = dict(sorted(data_dict.items()))
landing_sites = list(sorted_dict.values())[:10]

for i, site in enumerate(landing_sites):
    print(f"{i+1}: {site}")
    print("-"*50)