import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import pickle
from torch.utils.data import Dataset, DataLoader

import random

to_pil_image = transforms.ToPILImage()

def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('out/generated_images.gif', imgs)
    
def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"out/output{epoch}.jpg")
    
def save_loss_plot(train_loss, valid_loss):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('out/loss.jpg')
    plt.show()
    
    
class ImageDataset(Dataset):
    """ A dataset class for the images captured in the CarRacing Env"""
    
    def __init__(self, filename):
        """
        Args:
            filename (string): Path to the pickle file containing our
                images as (3, 64, 64) tensors
        """
        
        file = open(filename, "rb")
        self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): the index of the desired element
        """
        
        return self.data[idx], self.data[idx]
        