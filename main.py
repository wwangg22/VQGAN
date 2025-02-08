from model import VQGAN
import torch
import torch.optim as optim
import numpy as np
from utils import PNGImageDataset, compute_dataset_variance
from torch.utils.data import DataLoader
import torchvision.transforms as T
import cv2

transform = T.Compose([
    T.ToTensor()           # scales [0, 255] -> [0, 1] for each channel
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQGAN(h_dim=512, n_embeddings=32, embedding_dim=32, scale_factor=4).to(device)
# model.load_model("model.pth")

gen_optimizer = optim.Adam(
    list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.vector_quantization.parameters()),
    lr=2e-4,
    amsgrad=True
)
discrim_optimizer = optim.Adam(
    model.discriminator.parameters(),
    lr=2e-4,
    amsgrad=True
)
model.train()

n_updates = 10000

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

log_interval = 10
save = True
train_dataset = PNGImageDataset(folder_path='./images', transform=transform)
x_train_var = 1.0
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)


def train():
    update_count = 0
    while update_count < n_updates:
        for x in train_loader:
            # x is a batch of images with shape (batch_size, 3, height, width)
            x = x.to(device)
            if update_count % 2:
                gen_optimizer.zero_grad()
            else:
                discrim_optimizer.zero_grad()

            # Forward pass: run through VQ-VAE
            loss, x_hat, perplexity = model(x)
            
            # Backprop
            loss.backward()
            if update_count % 2:
                gen_optimizer.step()
            else:
                discrim_optimizer.step()

            # Record results
            results["perplexities"].append(perplexity.item())
            results["loss_vals"].append(loss.item())
            results["n_updates"] = update_count

            # Logging
            if update_count % log_interval == 0:
                #display xhat
                x_hat = x_hat[0].detach().cpu().numpy()
                # x_hat = np.transpose(x_hat, (0, 2, 3, 1))
                x_hat_img = np.transpose(x_hat, (1, 2, 0))

                 # If the output is in [0,1], scale it to [0,255]
                x_hat_img = np.clip(x_hat_img, 0, 1)  # ensure values are within [0,1]
                x_hat_img = (x_hat_img * 255).astype(np.uint8)
                x_hat_img = cv2.cvtColor(x_hat_img, cv2.COLOR_RGB2BGR)

                cv2.imshow('xhat', x_hat_img)
                cv2.waitKey(1)

                if save:
                    model.save_model("model.pth")  # or your chosen file path

                print(
                    f'Update #{update_count} | '
                    f'Recon Error: {np.mean(results["recon_errors"][-log_interval:]):.4f} | '
                    f'Loss: {np.mean(results["loss_vals"][-log_interval:]):.4f} | '
                    f'Perplexity: {np.mean(results["perplexities"][-log_interval:]):.4f}'
                )

            update_count += 1
            if update_count >= n_updates:
                break  # Stop if we've reached the desired number of updates

    return results

if __name__ == "__main__":
    train()