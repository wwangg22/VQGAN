from model import VQGAN
import torch
import torch.optim as optim
import numpy as np
from utils import PNGImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import cv2
import sys

transform = T.Compose([
    T.Resize((256, 256)),  
    T.ToTensor(), 
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQGAN(h_dim=468, n_embeddings=64, embedding_dim=32, scale_factor=5, dthreshold=0).to(device)

# Load the model
model.load_model("model.pth")

gen_optimizer = optim.Adam(
    list(model.encoder.parameters()) + 
    list(model.decoder.parameters()) + 
    list(model.vector_quantization.parameters()),
    lr=2e-4,
    amsgrad=True
)
discrim_optimizer = optim.Adam(
    model.discriminator.parameters(),
    lr=2e-4,
    amsgrad=True
)
model.train()

n_updates = 1000000

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

log_interval = 10
save = True
train_dataset = PNGImageDataset(folder_path='./output_images/train_f', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)

def train():
    update_count = 0
    try:
        while update_count < n_updates:
            for x in train_loader:
                x = x.to(device)
                
                # Even/odd logic for generator/discriminator updates
                if update_count % 2:
                    gen_optimizer.zero_grad()
                    loss, x_hat, perplexity = model(x, gen_update=True)
                else:
                    discrim_optimizer.zero_grad()
                    loss, x_hat, perplexity = model(x, gen_update=False)

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
                    # Convert the first reconstructed image to numpy and save/show
                    x_hat_np = x_hat[0].detach().cpu().numpy()
                    x_hat_img = np.transpose(x_hat_np, (1, 2, 0))
                    x_hat_img = np.clip(x_hat_img, 0, 1)
                    x_hat_img = (x_hat_img * 255).astype(np.uint8)
                    x_hat_img = cv2.cvtColor(x_hat_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('xhat_debug.png', x_hat_img)

                    if save:
                        model.save_model("model.pth")

                    print(
                        f'Update #{update_count} | '
                        f'Loss: {np.mean(results["loss_vals"][-log_interval:]):.4f} | '
                        f'Perplexity: {np.mean(results["perplexities"][-log_interval:]):.4f}'
                    )

                update_count += 1
                if update_count >= n_updates:
                    break  # Stop if we've reached the desired number of updates
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model before exiting...")
        model.save_model("model.pth")
        # Optionally save any other logs or do other cleanup tasks here.
        # sys.exit(0)  # Optionally force the script to exit here
    return results

if __name__ == "__main__":
    train()
