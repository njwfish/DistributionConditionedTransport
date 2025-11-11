import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST


class MNISTColorsDataset(Dataset):
    """Dataset for MNIST digits with continuous RGB color transformations.
    
    Each sample randomly selects a digit class and color for source and target,
    then samples images and applies the color transformation on-the-fly.
    """
    
    def __init__(
            self, 
            n_sets: int = 10_000, 
            set_size: int = 100, 
            digit_class: Optional[int] = None,  # If None, use all digits
            seed: Optional[int] = None,
            data_root: str = './data',
            train: bool = True,
            data_shape: Tuple[int, int, int] = (3, 28, 28),
            ):
        """
        Args:
            n_sets: Number of random colors to pre-generate
            set_size: Number of samples per set
            digit_class: Specific digit class to use (0-9), or None for all digits
            seed: Random seed for reproducibility
            data_root: Root directory for MNIST data
            train: Whether to use training or test set
        """
        self.n_sets = n_sets
        self.set_size = set_size
        self.digit_class = digit_class
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        
        self.mnist_dataset = MNIST(
            root=data_root, 
            train=train, 
            download=True, 
            transform=transform
        )
        
        # Organize MNIST data by digit class
        self.digit_indices = {}
        for i in range(len(self.mnist_dataset)):
            _, label = self.mnist_dataset[i]
            if label not in self.digit_indices:
                self.digit_indices[label] = []
            self.digit_indices[label].append(i)
            
        # Convert to numpy arrays for faster sampling
        for digit in self.digit_indices:
            self.digit_indices[digit] = np.array(self.digit_indices[digit])
        
        # Pre-generate n_sets random RGB colors
        self.colors = torch.rand(n_sets, 3)  # (n_sets, 3)
        
    def _apply_color_transform(self, mnist_images, color):
        """Apply a single RGB color to a batch of grayscale MNIST images.
        
        Args:
            mnist_images: Batch of grayscale MNIST image tensors (batch_size, 1, 28, 28)
            color: RGB color tensor (3,)
            
        Returns:
            Colored image tensors (batch_size, 3, 28, 28)
        """
        # Remove channel dimension: (batch_size, 28, 28)
        gray_imgs = mnist_images.squeeze(1)
        
        # Expand color for broadcasting: (1, 3, 1, 1)
        color_expanded = color.view(1, 3, 1, 1)
        
        # Expand grayscale to 3 channels: (batch_size, 3, 28, 28)
        gray_expanded = gray_imgs.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Apply color transformation via broadcasting
        colored_imgs = gray_expanded * color_expanded
        
        return colored_imgs
    
    def _sample_class_images(self, digit_class):
        """Sample set_size images from a specific digit class.
        
        Args:
            digit_class: The digit class to sample from (0-9)
            
        Returns:
            Batch of grayscale images (set_size, 1, 28, 28)
        """
        digit_indices = self.digit_indices[digit_class]
        
        if len(digit_indices) < self.set_size:
            # Sample with replacement if not enough samples
            selected_indices = np.random.choice(digit_indices, size=self.set_size, replace=True)
        else:
            # Sample without replacement
            selected_indices = np.random.choice(digit_indices, size=self.set_size, replace=False)
        
        # Load images
        mnist_images = []
        for mnist_idx in selected_indices:
            mnist_img, _ = self.mnist_dataset[mnist_idx]
            mnist_images.append(mnist_img)
        
        # Stack into batch tensor: (set_size, 1, 28, 28)
        return torch.stack(mnist_images)
        
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        """Get a pair of source and target samples with random classes and colors."""
        # Select random classes
        if self.digit_class is not None:
            source_class = target_class = self.digit_class
        else:
            source_class = np.random.randint(0, 10)
            target_class = np.random.randint(0, 10)
        
        # Select random colors (use idx for source, random for target)
        source_color_idx = idx
        target_color_idx = np.random.randint(0, self.n_sets)
        
        # Sample images from classes
        source_images = self._sample_class_images(source_class)
        target_images = self._sample_class_images(target_class)
        
        # Apply colors
        source_samples = self._apply_color_transform(source_images, self.colors[source_color_idx])
        target_samples = self._apply_color_transform(target_images, self.colors[target_color_idx])
        
        return {
            'source_samples': source_samples,
            'target_samples': target_samples,
            'source_idx': source_color_idx,
            'target_idx': target_color_idx
        }


class FashionMNISTColorsDataset(Dataset):
    """Dataset for Fashion-MNIST items with continuous RGB color transformations.
    
    Each sample randomly selects an item class and color for source and target,
    then samples images and applies the color transformation on-the-fly.
    
    Fashion-MNIST classes:
    0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
    5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    """
    
    def __init__(
            self, 
            n_sets: int = 10_000, 
            set_size: int = 100, 
            item_class: Optional[int] = None,  # If None, use all classes
            seed: Optional[int] = None,
            data_root: str = './data',
            train: bool = True,
            ):
        """
        Args:
            n_sets: Number of random colors to pre-generate
            set_size: Number of samples per set
            item_class: Specific item class to use (0-9), or None for all classes
            seed: Random seed for reproducibility
            data_root: Root directory for Fashion-MNIST data
            train: Whether to use training or test set
        """
        self.n_sets = n_sets
        self.set_size = set_size
        self.item_class = item_class
        
        # Class names for Fashion-MNIST
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Load Fashion-MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        
        self.fashion_dataset = FashionMNIST(
            root=data_root, 
            train=train, 
            download=True, 
            transform=transform
        )
        
        # Organize Fashion-MNIST data by item class
        self.item_indices = {}
        for i in range(len(self.fashion_dataset)):
            _, label = self.fashion_dataset[i]
            if label not in self.item_indices:
                self.item_indices[label] = []
            self.item_indices[label].append(i)
            
        # Convert to numpy arrays for faster sampling
        for item in self.item_indices:
            self.item_indices[item] = np.array(self.item_indices[item])
        
        # Pre-generate n_sets random RGB colors
        self.colors = torch.rand(n_sets, 3)  # (n_sets, 3)
        
    def _apply_color_transform(self, fashion_images, color):
        """Apply a single RGB color to a batch of grayscale Fashion-MNIST images.
        
        Args:
            fashion_images: Batch of grayscale Fashion-MNIST image tensors (batch_size, 1, 28, 28)
            color: RGB color tensor (3,)
            
        Returns:
            Colored image tensors (batch_size, 3, 28, 28)
        """
        # Remove channel dimension: (batch_size, 28, 28)
        gray_imgs = fashion_images.squeeze(1)
        
        # Expand color for broadcasting: (1, 3, 1, 1)
        color_expanded = color.view(1, 3, 1, 1)
        
        # Expand grayscale to 3 channels: (batch_size, 3, 28, 28)
        gray_expanded = gray_imgs.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Apply color transformation via broadcasting
        colored_imgs = gray_expanded * color_expanded
        
        return colored_imgs
    
    def _sample_class_images(self, item_class):
        """Sample set_size images from a specific item class.
        
        Args:
            item_class: The item class to sample from (0-9)
            
        Returns:
            Batch of grayscale images (set_size, 1, 28, 28)
        """
        class_indices = self.item_indices[item_class]
        
        if len(class_indices) < self.set_size:
            # Sample with replacement if not enough samples
            selected_indices = np.random.choice(class_indices, size=self.set_size, replace=True)
        else:
            # Sample without replacement
            selected_indices = np.random.choice(class_indices, size=self.set_size, replace=False)
        
        # Load images
        fashion_images = []
        for fashion_idx in selected_indices:
            fashion_img, _ = self.fashion_dataset[fashion_idx]
            fashion_images.append(fashion_img)
        
        # Stack into batch tensor: (set_size, 1, 28, 28)
        return torch.stack(fashion_images)
        
    def get_class_name(self, class_idx):
        """Get the name of a Fashion-MNIST class."""
        return self.class_names[class_idx]
        
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        """Get a pair of source and target samples with random classes and colors."""
        # Select random classes
        if self.item_class is not None:
            source_class = target_class = self.item_class
        else:
            source_class = np.random.randint(0, 10)
            target_class = np.random.randint(0, 10)
        
        # Select random colors (use idx for source, random for target)
        source_color_idx = idx
        target_color_idx = np.random.randint(0, self.n_sets)
        
        # Sample images from classes
        source_images = self._sample_class_images(source_class)
        target_images = self._sample_class_images(target_class)
        
        # Apply colors
        source_samples = self._apply_color_transform(source_images, self.colors[source_color_idx])
        target_samples = self._apply_color_transform(target_images, self.colors[target_color_idx])
        
        return {
            'source_samples': source_samples,
            'target_samples': target_samples,
            'source_idx': source_color_idx,
            'target_idx': target_color_idx
        }


# Convenience functions to create datasets with specific configurations
def create_mnist_colors_dataset(
    digit_class: Optional[int] = None,
    n_sets: int = 1000,
    set_size: int = 100,
    seed: Optional[int] = 42,
    **kwargs
):
    """Create an MNIST colors dataset with common configurations."""
    return MNISTColorsDataset(
        n_sets=n_sets,
        set_size=set_size,
        digit_class=digit_class,
        seed=seed,
        **kwargs
    )


def create_fashion_mnist_colors_dataset(
    item_class: Optional[int] = None,
    n_sets: int = 1000,
    set_size: int = 100,
    seed: Optional[int] = 42,
    **kwargs
):
    """Create a Fashion-MNIST colors dataset with common configurations."""
    return FashionMNISTColorsDataset(
        n_sets=n_sets,
        set_size=set_size,
        item_class=item_class,
        seed=seed,
        **kwargs
    )