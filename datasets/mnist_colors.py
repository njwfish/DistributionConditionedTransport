import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST


class MNISTColorsDataset(Dataset):
    """Dataset for MNIST digits with continuous RGB color transformations.
    
    Each set contains MNIST digits of the same class with random RGB colors.
    Source and target samples use the same reservoir with random target sampling.
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
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
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
        
        # Pre-organize MNIST data by digit class for faster access
        self.digit_indices = {}
        for i in range(len(self.mnist_dataset)):
            _, label = self.mnist_dataset[i]
            if label not in self.digit_indices:
                self.digit_indices[label] = []
            self.digit_indices[label].append(i)
            
        # Convert to numpy arrays for faster sampling
        for digit in self.digit_indices:
            self.digit_indices[digit] = np.array(self.digit_indices[digit])
            
        if digit_class is not None:
            self.mnist_indices = self.digit_indices[digit_class]
        else:
            self.mnist_indices = list(range(len(self.mnist_dataset)))
            
        # Pre-generate data for all sets (single reservoir)
        self.data = self._generate_all_sets()
        
    def _apply_color_transform_batch(self, mnist_images):
        """Apply continuous RGB color transformation to a batch of grayscale MNIST images.
        
        Args:
            mnist_images: Batch of grayscale MNIST image tensors (batch_size, 1, 28, 28)
            
        Returns:
            Colored image tensors (batch_size, 3, 28, 28)
        """
        batch_size = mnist_images.shape[0]
        
        # Remove channel dimension: (batch_size, 28, 28)
        gray_imgs = mnist_images.squeeze(1)
        
        # Sample RGB colors for each image: (batch_size, 3)
        rgb_colors = torch.rand(batch_size, 3)
        
        # Expand dimensions for broadcasting: (batch_size, 3, 1, 1)
        rgb_colors = rgb_colors.unsqueeze(-1).unsqueeze(-1)
        
        # Expand grayscale to 3 channels: (batch_size, 3, 28, 28)
        gray_expanded = gray_imgs.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Apply color transformation via broadcasting
        colored_imgs = gray_expanded * rgb_colors
        
        return colored_imgs
        
    def _generate_set(self, set_idx):
        """Generate a single set of colored MNIST digits."""
        # Choose a random digit class for this set (if not fixed)
        if self.digit_class is not None:
            target_digit = self.digit_class
        else:
            target_digit = np.random.randint(0, 10)
            
        # Get pre-organized indices for this digit
        digit_indices = self.digit_indices[target_digit]
        
        if len(digit_indices) < self.set_size:
            # If not enough samples, sample with replacement
            selected_indices = np.random.choice(digit_indices, size=self.set_size, replace=True)
        else:
            # Sample without replacement
            selected_indices = np.random.choice(digit_indices, size=self.set_size, replace=False)
            
        # Load all images at once
        mnist_images = []
        for mnist_idx in selected_indices:
            mnist_img, _ = self.mnist_dataset[mnist_idx]
            mnist_images.append(mnist_img)
            
        # Stack into batch tensor: (set_size, 1, 28, 28)
        mnist_batch = torch.stack(mnist_images)
        
        # Apply color transformation to entire batch
        colored_tensor = self._apply_color_transform_batch(mnist_batch)
        
        return colored_tensor
        
    def _generate_all_sets(self):
        """Pre-generate all sets of data as a single reservoir using batch processing."""
        print(f"Generating {self.n_sets} sets of colored MNIST data...")
        
        # Process in batches for better memory efficiency
        batch_size = 100  # Process 100 sets at a time
        all_data = []
        
        for batch_start in range(0, self.n_sets, batch_size):
            batch_end = min(batch_start + batch_size, self.n_sets)
            current_batch_size = batch_end - batch_start
            
            if batch_start % 1000 == 0:
                print(f"Generated {batch_start}/{self.n_sets} sets")
            
            # Generate batch of sets
            batch_data = []
            for set_idx in range(batch_start, batch_end):
                data_set = self._generate_set(set_idx)
                batch_data.append(data_set)
            
            # Stack batch and add to all_data
            if batch_data:
                batch_tensor = torch.stack(batch_data)
                all_data.append(batch_tensor)
            
        # Stack all batches - single reservoir
        data = torch.cat(all_data, dim=0)  # (n_sets, set_size, 3, 28, 28)
        
        print(f"Generated data shape: {data.shape}")
        
        return data
        
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        """Get a pair of source and target samples."""
        source_idx = idx
        # Sample a random target index from the same reservoir
        target_idx = np.random.choice(self.n_sets)
        
        return {
            'source_samples': self.data[source_idx],
            'target_samples': self.data[target_idx],
            'source_idx': source_idx,
            'target_idx': target_idx
        }


class FashionMNISTColorsDataset(Dataset):
    """Dataset for Fashion-MNIST items with continuous RGB color transformations.
    
    Each set contains Fashion-MNIST items of the same class with random RGB colors.
    Source and target samples use the same reservoir with random target sampling.
    
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
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
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
        
        # Pre-organize Fashion-MNIST data by item class for faster access
        self.item_indices = {}
        for i in range(len(self.fashion_dataset)):
            _, label = self.fashion_dataset[i]
            if label not in self.item_indices:
                self.item_indices[label] = []
            self.item_indices[label].append(i)
            
        # Convert to numpy arrays for faster sampling
        for item in self.item_indices:
            self.item_indices[item] = np.array(self.item_indices[item])
            
        if item_class is not None:
            self.fashion_indices = self.item_indices[item_class]
        else:
            self.fashion_indices = list(range(len(self.fashion_dataset)))
            
        # Pre-generate data for all sets (single reservoir)
        self.data = self._generate_all_sets()
        
    def _apply_color_transform_batch(self, fashion_images):
        """Apply continuous RGB color transformation to a batch of grayscale Fashion-MNIST images.
        
        Args:
            fashion_images: Batch of grayscale Fashion-MNIST image tensors (batch_size, 1, 28, 28)
            
        Returns:
            Colored image tensors (batch_size, 3, 28, 28)
        """
        batch_size = fashion_images.shape[0]
        
        # Remove channel dimension: (batch_size, 28, 28)
        gray_imgs = fashion_images.squeeze(1)
        
        # Sample RGB colors for each image: (batch_size, 3)
        rgb_colors = torch.rand(batch_size, 3)
        
        # Expand dimensions for broadcasting: (batch_size, 3, 1, 1)
        rgb_colors = rgb_colors.unsqueeze(-1).unsqueeze(-1)
        
        # Expand grayscale to 3 channels: (batch_size, 3, 28, 28)
        gray_expanded = gray_imgs.unsqueeze(1).expand(-1, 3, -1, -1)
        
        # Apply color transformation via broadcasting
        colored_imgs = gray_expanded * rgb_colors
        
        return colored_imgs
        
    def _generate_set(self, set_idx):
        """Generate a single set of colored Fashion-MNIST items."""
        # Choose a random item class for this set (if not fixed)
        if self.item_class is not None:
            target_class = self.item_class
        else:
            target_class = np.random.randint(0, 10)
            
        # Get pre-organized indices for this class
        class_indices = self.item_indices[target_class]
        
        if len(class_indices) < self.set_size:
            # If not enough samples, sample with replacement
            selected_indices = np.random.choice(class_indices, size=self.set_size, replace=True)
        else:
            # Sample without replacement
            selected_indices = np.random.choice(class_indices, size=self.set_size, replace=False)
            
        # Load all images at once
        fashion_images = []
        for fashion_idx in selected_indices:
            fashion_img, _ = self.fashion_dataset[fashion_idx]
            fashion_images.append(fashion_img)
            
        # Stack into batch tensor: (set_size, 1, 28, 28)
        fashion_batch = torch.stack(fashion_images)
        
        # Apply color transformation to entire batch
        colored_tensor = self._apply_color_transform_batch(fashion_batch)
        
        return colored_tensor
        
    def _generate_all_sets(self):
        """Pre-generate all sets of data as a single reservoir using batch processing."""
        print(f"Generating {self.n_sets} sets of colored Fashion-MNIST data...")
        
        # Process in batches for better memory efficiency
        batch_size = 100  # Process 100 sets at a time
        all_data = []
        
        for batch_start in range(0, self.n_sets, batch_size):
            batch_end = min(batch_start + batch_size, self.n_sets)
            current_batch_size = batch_end - batch_start
            
            if batch_start % 1000 == 0:
                print(f"Generated {batch_start}/{self.n_sets} sets")
            
            # Generate batch of sets
            batch_data = []
            for set_idx in range(batch_start, batch_end):
                data_set = self._generate_set(set_idx)
                batch_data.append(data_set)
            
            # Stack batch and add to all_data
            if batch_data:
                batch_tensor = torch.stack(batch_data)
                all_data.append(batch_tensor)
            
        # Stack all batches - single reservoir
        data = torch.cat(all_data, dim=0)  # (n_sets, set_size, 3, 28, 28)
        
        print(f"Generated data shape: {data.shape}")
        
        return data
        
    def get_class_name(self, class_idx):
        """Get the name of a Fashion-MNIST class."""
        return self.class_names[class_idx]
        
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        """Get a pair of source and target samples."""
        source_idx = idx
        # Sample a random target index from the same reservoir
        target_idx = np.random.choice(self.n_sets)
        
        return {
            'source_samples': self.data[source_idx],
            'target_samples': self.data[target_idx],
            'source_idx': source_idx,
            'target_idx': target_idx
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