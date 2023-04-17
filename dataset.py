from pathlib import Path
import torch


class OODDataset:

    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        # The root directory should be structure as follows
        # Images, Mask
        # or Images
        dirs = list(self.root.iterdir())
        if len(dirs) == 1:
            self.x = list((dirs / 'imags').iterdir())
            self.y = torch.ones(len(self.x))
            self.has_mask = False
        else:
            self.x = list((dirs / 'imags').iterdir())
            self.y = [f.replace('images', 'masks') for f in self.x]
            self.has_mask = True
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
