import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(123)

# Simplified hyperparameters
latent_dim = 64  # Reduced from 100
hidden_dim = 256  # Single hidden dimension instead of multiple increasing sizes
image_size = 28  # Native MNIST size instead of resizing to 64x64
batch_size = 32  # Smaller batch size
lr = 0.0002
num_epochs = 10  # Fewer epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simplified data preprocessing - No resizing needed
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load the MNIST dataset
dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # No workers for simplicity

# Simplified Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # Input is latent vector z
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, image_size * image_size),
            nn.Tanh()  # Output values between -1 and 1
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, image_size, image_size)
        return img

# Simplified Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input is an image
            nn.Flatten(),
            nn.Linear(image_size * image_size, hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Simplified optimizers (no beta parameters)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Simplified training function
def train_gan():
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    images = []
    
    print("Starting Training Loop...")
    # Save initial random output
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        sample_imgs = generator(z).detach().cpu()
        images.append(torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True))
    
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            # Configure input
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Create labels for real and fake data
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss on real images
            real_pred = discriminator(real_imgs)
            d_loss_real = adversarial_loss(real_pred, real_label)
            
            # Loss on fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs.detach())
            d_loss_fake = adversarial_loss(fake_pred, fake_label)
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generate images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            fake_pred = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_pred, real_label)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Print progress less frequently
            if i % 500 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
            
            # Save losses for plotting
            if i % 100 == 0:
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())
        
        # Save generated images at the end of each epoch
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            sample_imgs = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)
            images.append(grid)
            
        print(f"Epoch {epoch} complete")
    
    return G_losses, D_losses, images

# Simplified visualization function
def visualize_results(G_losses, D_losses, images):
    # Plot losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations (x100)")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("gan_loss_plot.png")
    plt.close()
    
    # Plot generated images
    plt.figure(figsize=(12,6))
    for i, img in enumerate(images[:10]):  # Show only up to 10 images
        plt.subplot(2, 5, i+1)
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(f"Epoch {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("gan_generated_images.png")
    plt.close()

# Main execution
def main():
    # Train the GAN
    print(f"Training on {device}")
    print(f"Total epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    G_losses, D_losses, images = train_gan()
    
    # Visualize training results
    visualize_results(G_losses, D_losses, images)
    
    # Generate a few final images
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        gen_imgs = generator(z).detach().cpu()
        
        plt.figure(figsize=(8,8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(gen_imgs[i, 0, :, :], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("final_generated_images.png")
        plt.close()
    
    print("GAN training complete!")

if __name__ == "__main__":
    main()
