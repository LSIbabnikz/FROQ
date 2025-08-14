
import os

from PIL import Image

from torch.utils.data import Dataset

class DummyDataset(Dataset):

    def __init__(
            self,
            trans,
            loc):
        super().__init__()

        self.items = []
        for (dir, subdirs, files) in os.walk(loc):
            self.items.extend(list(map(lambda x: os.path.join(dir, x), files)))
        self.trans = trans
    
    def __getitem__(self, x):
        image_loc = self.items[x]
        img = self.trans(Image.open(image_loc).convert("RGB"))
        return image_loc, img
    
    def __len__(self):
        return len(self.items)