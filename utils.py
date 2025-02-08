
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PNGImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        super().__init__()
        self.folder_path = folder_path
        # Collect all PNG file paths
        self.image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def compute_dataset_variance(dataset, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_pixels = []
    for batch in loader:
        # batch shape: (batch_size, channels, height, width)
        # flatten to (batch_size, -1)
        pixels = batch.view(batch.size(0), -1).float()
        all_pixels.append(pixels)
    
    # Concatenate everything into one big tensor
    all_pixels = torch.cat(all_pixels, dim=0)
    var = all_pixels.var().item()  # or use .mean(dim=1).var(dim=0) as needed
    return var