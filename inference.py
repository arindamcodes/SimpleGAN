import torch
import torch.nn as nn
import matplotlib.pyplot as plt

batch_size = 1
z_dim = 64
img_dim = 28 * 28
device = "cpu"

class Generator(nn.Module):

    def __init__(self, z_dim, img_dim):

        super().__init__()
        self.fc = nn.Linear(z_dim, 256)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(256, img_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

gen = Generator(z_dim, img_dim)
gen.load_state_dict(torch.load("gan_after_training.pt"))
gen = gen.to(device)

# Input to the generator
fixed_noise = torch.randn((batch_size, z_dim))

with torch.no_grad():
    generated = gen(fixed_noise).reshape(-1, 1, 28, 28)
    generated = generated.squeeze() 
    generated = generated.numpy()
    plt.imshow(generated, cmap='gray')  
    plt.axis('off')
    plt.show()







