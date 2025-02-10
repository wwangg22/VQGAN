import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from dis_model import NLayerDiscriminator, weights_init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, scale_factor=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1)
        )
        layers = []
        # --- 1) Initial conv (stride=1) to go from RGB -> hidden_channels ---
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # --- 2) Downsample scale_factor times ---
        for _ in range(scale_factor):
            layers.append(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))

        # --- 3) Optional final conv or res-block at stride=1 ---
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))

        self.conv = nn.Sequential(*layers)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, hidden_channels // 4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channels // 4, hidden_channels//2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        # )

    def forward(self, x):

        x = self.conv(x)

        x = x + self.res_block(x)

        x = x + self.res_block(x)

        return x

class Decoder(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, scale_factor=1):
        super().__init__()

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        )
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1)
        )
        layers = []
        # --- 1) Optional initial conv (stride=1) to process latent features ---
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))

        # --- 2) Upsample scale_factor times ---
        for _ in range(scale_factor):
            layers.append(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))

        # --- 3) Final conv to map hidden_channels -> out_channels (RGB) ---
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1))

        self.conv = nn.Sequential(*layers)

        # self.conv = nn.Sequential(
        #     nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(hidden_channels // 2, out_channels=hidden_channels//4, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(hidden_channels // 4, out_channels, kernel_size=4, stride =2, padding=1),
        # )

    def forward(self, x ):
        x = self.initial(x)

        x = x + self.res_block(x)

        x = x + self.res_block(x)

        x = self.conv(x)

        return x
    

#https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta = 0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # first torch.sum produces (B*H*W,1) tensor, second torch.sum produces (n_e,) tensor, according to 
        #broadcasting rules, produces (B*H*W, n_e) tensor
        # matmul between (B*H*W, e_dim) and (e_dim, n_e) produces (B*H*W, n_e) tensor)
        #outputs a tensor of shape (B*H*W, n_e)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients 
        #pass through basically
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices




class VQGAN(nn.Module):
    def __init__(self, h_dim,
                 n_embeddings, embedding_dim, scale_factor=1, 
                 beta=0.25, save_img_embedding_map=False, disc_in_channels=3,
                 disc_num_layers=3, use_actnorm = False, disc_ndf=64,
                 embweight=1.0, pweight=1.0, dweight=1.0, 
                 dthreshold=5000):
        super().__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(in_channels=3, hidden_channels=h_dim, scale_factor=scale_factor)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(in_channels=embedding_dim, hidden_channels=h_dim, out_channels=3, scale_factor=scale_factor)
        self.embweight = embweight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)

        self.ploss = LPIPS().eval()
        self.pweight = pweight
        self.dweight = dweight

        self.step = 0
        self.dthreshold = dthreshold

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None
    
    def get_last_layer(self):
        return self.decoder.conv[-1]

    def forward(self, x, verbose=False, gen_update=True):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)
        #l1 distance
        rec_loss = torch.abs(x.contiguous() - x_hat.contiguous())
        #perceptual loss calculation using VGG12 at different conv layers
        if self.pweight > 0:
            p_loss = self.ploss(x.contiguous(), x_hat.contiguous())
            rec_loss = rec_loss + self.pweight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        # print('nll_loss:', nll_loss)

        if gen_update:
            #this update is for the generator to fool the discriminator
            logits_fake = self.discriminator(x_hat.contiguous())
            g_loss = -torch.mean(logits_fake)
            #normalize the loss of the discriminiator so it does not overpower gradient update to VAE
            nll_grads = torch.autograd.grad(nll_loss, self.get_last_layer().weight, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.get_last_layer().weight, retain_graph=True)[0]
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
            d_weight = d_weight * self.dweight
            #zero out the gradient of the discriminator for the first n steps
            disc_factor = 1.0
            if self.step < self.dthreshold:
                disc_factor = 0.0
            # print('disc_factor:', disc_factor)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.embweight * embedding_loss.mean()
            # print("embedding_loss:", embedding_loss.mean())
            self.step+=1
        else:
            #this update is for the discriminator
            logits_real = self.discriminator(x.contiguous().detach())
            logits_fake = self.discriminator(x_hat.contiguous().detach())
            #zero out the gradient of the discriminator for the first n steps
            disc_factor = 1.0
            if self.step < self.dthreshold:
                disc_factor = 0.0
            #hinge loss 
            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            d_loss = 0.5 * (loss_real + loss_fake)

            loss = disc_factor * d_loss
        
        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return loss, x_hat, perplexity
    
    def save_model(self, file_path: str):
        """
        Save the model's state_dict to the specified file path.

        Args:
            file_path (str): Where to save the model (.pth or .pt file).
        """
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path:str):
        self.load_state_dict(torch.load(file_path))
        print(f'model loaded from {file_path}')




