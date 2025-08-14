
import os

from torch.utils.data import Dataset

# Baseline Dataset class
class Glint360KSubset_Base(Dataset):

    def __init__(
            self, 
            trans,
            loc):

        super().__init__()

        self.items = []
        for (dir, subdirs, files) in os.walk(loc):
            self.items.extend(list(filter(lambda x: x.endswith("jpg"), map(lambda x: os.path.join(dir,x), files))))
        self.trans = trans
    
    def __len__(self):
        return len(self.items)
