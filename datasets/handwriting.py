from torch.utils.data import Dataset
from torchvision import transforms
import os, glob, cv2, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
import tqdm

from typing import Optional, Tuple
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Set this depending on your CPU; 8–32 is common
MAX_WORKERS = os.cpu_count() or 8
cv2.setNumThreads(0)  # avoid oversubscription with our own thread pool

def read_img_cv(path, grayscale=True, size=None):
    # grayscale=True is typical for character images
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    img = cv2.imread(path, flag)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    if size is not None and size != (img.shape[1], img.shape[0]):
        # INTER_AREA is best for downsampling (128->28)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img  # np.ndarray


import torch.nn.functional as F
import numpy as np

def resize_batch(arr_np, size=(28, 28)):
    """
    arr_np: np.uint8 array of shape (N, C, H, W)
    returns: torch.FloatTensor (N, C, 28, 28)
    """
    t = torch.from_numpy(arr_np)
    return F.interpolate(t, size=size, mode="bilinear", antialias=True)


class HandwritingDataset(Dataset):
    """Dataset for handwriting data with continuous RGB color transformations.
    
    Each sample randomly selects a handwriting class and color for source and target,
    then samples images and applies the color transformation on-the-fly.
    """
    def __init__(
            self, 
            set_size: int = 64, 
            n_sets: Optional[int] = None,
            seed: Optional[int] = None,
            data_root: str = '/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/CoupledDistributionEmbeddings/data',
            train: bool = True,
            data_shape: Tuple[int, int, int] = (1, 28, 28),
            train_split: float = 0.9,
            frac_unique_writers: Optional[float] = None,
            n_unique_writers: Optional[int] = None,
        ):
        """
        Args:
            n_sets: Number of random colors to pre-generate
            set_size: Number of samples per set
            seed: Random seed for reproducibility
            data_root: Root directory for MNIST data
            train: Whether to use training or test set
            frac_unique_writers: Fraction of unique writers to use (0.0-1.0)
            n_unique_writers: Absolute number of unique writers (for EmbeddingEncoder)
        """
        self.frac_unique_writers = frac_unique_writers
        self.n_unique_writers = n_unique_writers
        self.set_size = set_size
        self.data_shape = data_shape
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        by_writer_by_class = {}
        num_by_writer_by_class = {}
        paths_by_writer_by_class = {}

        i = 0
        if os.path.exists(f"{data_root}/by_write/X_uint8.npy"):
            X = np.load(f"{data_root}/by_write/X_uint8.npy", mmap_mode="r")        # zero-copy paging
            meta = np.load(f"{data_root}/by_write/meta.npz", allow_pickle=True)

            writers = meta["writers"].tolist()
            chars   = list(CHARS)
            starts  = meta["starts"]; stops = meta["stops"]
            wi_arr  = meta["writer_idx"]; ci_arr = meta["char_idx"]

            # Build nested dict of *views* (no data copies)
            by_writer_by_class = {w: {c: None for c in chars} for w in writers}
            for i in range(len(starts)):
                w = writers[int(wi_arr[i])]
                c = chars[int(ci_arr[i])]
                s, e = int(starts[i]), int(stops[i])
                by_writer_by_class[w][c] = X[s:e] / 255.0  # view into memmapped array

            num_by_writer_by_class = {
                w: {c: by_writer_by_class[w][c].shape[0] for c in chars} for w in writers
            }
        else:
            for writer_path in tqdm.tqdm(glob.glob(f"{data_root}/by_write/*/*")):
                writer = writer_path.split("/")[-1]
                by_writer_by_class[writer] = {}
                num_by_writer_by_class[writer] = {}
                paths_by_writer_by_class[writer] = {}

                for char in CHARS:
                    char_img_paths = glob.glob(f"{writer_path}/*/{char}/*")

                    # Threaded decode (much faster than serial)
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                        imgs = list(ex.map(lambda p: read_img_cv(p, grayscale=True), char_img_paths))

                    # Stack to (N, H, W) for grayscale or (N, H, W, C) if color
                    arr = np.stack(imgs, axis=0) if len(imgs) else np.empty((0, 1, 128, 128), dtype=np.uint8)
                    # expand to 1 channel in greyscale
                    arr = arr[:, np.newaxis, ...].repeat(self.data_shape[0], axis=1)

                    by_writer_by_class[writer][char] = 1 - arr / 255.0
                    num_by_writer_by_class[writer][char] = len(char_img_paths)
                    paths_by_writer_by_class[writer][char] = char_img_paths

                if n_sets and i > n_sets:
                    break
                i += 1

        self.by_writer_by_class = by_writer_by_class
        self.num_by_writer_by_class = num_by_writer_by_class
        self.paths_by_writer_by_class = paths_by_writer_by_class

        # split writers into train and test
        writers = np.array(list(by_writer_by_class.keys()))
        perm_idx = np.random.permutation(writers.shape[0])
        partition_idx = int(writers.shape[0] * train_split)
        self.train_writers, self.test_writers = writers[perm_idx[:partition_idx]], writers[perm_idx[partition_idx:]]

        # Optionally limit the number of unique writers by fraction or absolute count
        if self.frac_unique_writers is not None:
            n_to_use = max(1, int(len(self.train_writers) * self.frac_unique_writers))
            self.train_writers = self.train_writers[:n_to_use]
        elif self.n_unique_writers is not None:
            n_to_use = min(self.n_unique_writers, len(self.train_writers))
            self.train_writers = self.train_writers[:n_to_use]

        self.n_sets = len(self.train_writers)
        # Alias for EmbeddingEncoder compatibility
        self.n_unique_sets = self.n_sets

        # Build writer name to index mapping for embedding encoder (like MNIST set_idx)
        self.set_idx = np.arange(self.n_sets)  # numpy array for proper collation
        self.writer_to_idx = {w: i for i, w in enumerate(self.train_writers)}
        
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        """Get a pair of source and target samples with random classes and colors."""
        source_writer = np.random.choice(self.train_writers)
        target_writer = np.random.choice(self.train_writers)
        char_weights = np.array([min(self.num_by_writer_by_class[source_writer][char], self.num_by_writer_by_class[target_writer][char]) for char in CHARS])
        
        if char_weights.sum() == 0:
            # print("No characters to sample", self.num_by_writer_by_class[source_writer], self.num_by_writer_by_class[target_writer])
            return self.__getitem__(idx)
        
        char_weights = char_weights / char_weights.sum()
        
        chars_to_sample = np.random.choice(np.array(list(CHARS)), size=self.set_size, p=char_weights)

        source_samples = np.zeros((self.set_size, 1, 128, 128), dtype=np.float32)
        target_samples = np.zeros((self.set_size, 1, 128, 128), dtype=np.float32)
        vals, counts = np.unique(chars_to_sample, return_counts=True)
        i = 0
        labels = []
        for val, count in zip(vals, counts):
            source_char_img_idx = np.random.choice(np.arange(self.num_by_writer_by_class[source_writer][val]), size=count)
            target_char_img_idx = np.random.choice(np.arange(self.num_by_writer_by_class[target_writer][val]), size=count)
            source_samples[i:i+count] = self.by_writer_by_class[source_writer][val][source_char_img_idx]
            target_samples[i:i+count] = self.by_writer_by_class[target_writer][val][target_char_img_idx]
            labels.extend([val] * count)
            i += count

        return {
            'source_samples': resize_batch(source_samples, self.data_shape[1:]),
            'target_samples': resize_batch(target_samples, self.data_shape[1:]),
            'source_idx': self.set_idx[self.writer_to_idx[source_writer]],
            'target_idx': self.set_idx[self.writer_to_idx[target_writer]],
        }
