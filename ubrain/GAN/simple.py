import torch
import os
import tqdm


from nilearn import plotting
import pylab as plt

import numpy as np
import nibabel as nb
import pandas as pd



import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Définir le générateur
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(128, 3, kernel_size=4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Définir le discriminateur
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



# Exemple de boucle d'entraînement
def train_gan(generator, discriminator, dataset, epochs, batch_size):
    for epoch in range(epochs):
        for batch in dataset:
            # Redimensionner les images à 125x125
            real_images = resize_images(batch, new_size=(125, 125))
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Entraîner le discriminateur avec des images réelles
            optimizer_D.zero_grad()
            output = discriminator(real_images)
            errD_real = criterion(output, real_labels)
            errD_real.backward()

            # Générer des images factices
            noise = torch.randn(batch_size, 3, 125, 125)
            fake_images = generator(noise)

            # Entraîner le discriminateur avec des images factices
            output = discriminator(fake_images.detach())
            errD_fake = criterion(output, fake_labels)
            errD_fake.backward()
            optimizer_D.step()

            # Entraîner le générateur
            optimizer_G.zero_grad()
            output = discriminator(fake_images)
            errG = criterion(output, real_labels)
            errG.backward()
            optimizer_G.step()

        print(f"Epoch {epoch}, Discriminator Loss: {errD_real + errD_fake}, Generator Loss: {errG}")



if __name__ == "__main__" :
    path = "data/IXI-T2/"
    files = os.listdir(path)
    data_files = []
    for file in files :
        if ".nii.gz" in file :
            data_files.append(file)

    print("there is", len(data_files), "files detected")



    training_data = []
    for data in data_files :
        ni_file = nb.load(path + data)
        ni_data = ni_file.get_fdata().T
        for image in ni_data :
            training_data.append(image)
    

    print(len(training_data), "images extracted")

    # Créer les modèles
    generator = Generator()
    discriminator = Discriminator()

    # Définir les fonctions de perte et les optimiseurs
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Afficher les architectures
    print("Generator architecture:")
    print(generator)
    print("\nDiscriminator architecture:")
    print(discriminator)

    # Exemple d'utilisation
    train_gan(generator, discriminator, [training_data], epochs=10, batch_size=32)


