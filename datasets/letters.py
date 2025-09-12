import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from typing import Optional, List

all_letters = 'ABCDEFGHIJKLMNOPQRSTUVW'

class Letters(Dataset):
    def __init__(
        self, 
        data_shape: List[int],
        samples_per_letter_font: int = 1000,
        set_size: int = 100,
        noise_level: float = 0.1,
        train_letters: str = 'ABCDEFGHIJKLMNOPQRSTUVW',
        num_fonts: int = 10,
        seed: Optional[int] = 42,
        data_path='/orcd/archive/abugoot/001/Projects/njwfish/data/letters/'
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.size = samples_per_letter_font * num_fonts * len(train_letters)
        self.samples_per_letter_font = samples_per_letter_font
        self.set_size = set_size
        self.noise_level = noise_level
        self.train_letters = train_letters
        self.num_fonts = num_fonts
        self.data = {}
        # load image pngs from data_path
        for letter in all_letters:
            data_path_letter = os.path.join(data_path, letter)
            self.data[letter] = {}
            for font in range(num_fonts):
                data_path_letter_font = os.path.join(data_path_letter, f'{font}.png')
                self.data[letter][font] = {}
                if os.path.exists(data_path_letter_font):
                    im = self.im_to_2d_samples(Image.open(data_path_letter_font))
                    self.data[letter][font] = im
                    
                else:
                    raise FileNotFoundError(f'Font {font} not found for letter {letter}')

    def im_to_2d_samples(self, im):
        im = np.array(im)
        im_height, im_width = im.shape
        assert im_height == im_width, "Image must be square"
        probs = im.flatten() / im.sum()
        num_samples = self.samples_per_letter_font
        pixel_indices = np.random.choice(probs.shape[0], size=num_samples, p=probs)
        # convert pixel indices to 2d coordinates
        i, j = np.unravel_index(pixel_indices, (im_height, im_width))
        # sample uniform samples
        unif = np.random.uniform(0, 1 / im_height, size=(num_samples, 2))
        unif[:, 0] += j / im_height
        unif[:, 1] += i / im_width
        return unif.astype(np.float32)

    def __len__(self):
        return (self.num_fonts - 1) * len(self.train_letters)
    
    def __getitem__(self, idx):
        # map idx to letter and font
        letter = self.train_letters[idx // (self.num_fonts - 1)]
        font = idx % (self.num_fonts - 1)
        source_samples = self.data[letter][font]
        target_samples = self.data[letter][font + 1]
        # subsample to set size
        source_idx = np.random.choice(source_samples.shape[0], size=self.set_size, replace=False)
        target_idx = np.random.choice(target_samples.shape[0], size=self.set_size, replace=False)
    
        return {
            'source_samples': torch.from_numpy(source_samples[source_idx]),
            'target_samples': torch.from_numpy(target_samples[target_idx])
        }
            