import os
import tarfile
import io
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

class ShardedMammographyStreamer(IterableDataset):
    """
    Memory-Efficient Streaming Dataset for 500,000+ Images.
    Streams web-sharded .tar files (containing 1,000 images each) sequentially
    to keep RAM usage below 2 GB.
    """
    def __init__(self, shard_paths, transform=None):
        super(ShardedMammographyStreamer, self).__init__()
        self.shard_paths = shard_paths
        self.transform = transform or self.default_transform()

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_clahe(self, raw_bytes):
        """ Applies CLAHE contrast balancing to balance dense fibroglandular tissue """
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return None
        
        # Isolate breast contour (Otsu thresholding)
        _, thresh = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_gray = img_gray[y:y+h, x:x+w]
        
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    def parse_shard(self, shard_path):
        """ Stream-extracts samples without saving to local disk """
        with tarfile.open(shard_path, "r|*") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(('.png', '.jpg', '.dcm', '.jpeg')):
                    f = tar.extractfile(member)
                    if f is not None:
                        image_bytes = f.read()
                        pil_img = self.preprocess_clahe(image_bytes)
                        if pil_img is not None:
                            tensor = self.transform(pil_img)
                            # Extract metadata label encoded in filename (e.g., patient123_label1.png)
                            label = 1.0 if "malignant" in member.name.lower() or "label1" in member.name.lower() else 0.0
                            yield tensor, torch.tensor([label], dtype=torch.float32), member.name

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            for shard in self.shard_paths:
                yield from self.parse_shard(shard)
        else:
            # Multi-process GPU/CPU loader split
            per_worker = int(np.ceil(len(self.shard_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_shards = self.shard_paths[worker_id * per_worker:(worker_id + 1) * per_worker]
            for shard in iter_shards:
                yield from self.parse_shard(shard)

def create_streaming_loader(shard_folder, batch_size=64, num_workers=4):
    shard_paths = [os.path.join(shard_folder, f) for f in os.listdir(shard_folder) if f.endswith('.tar')]
    dataset = ShardedMammographyStreamer(shard_paths)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True
    )
