from torchvision import transforms
from torch.utils.data.dataset import Dataset


def transformations(transformations):
    return transforms.Compose(
        [
            transform for transform in transformations
        ]
    )

class ImageDataset(Dataset):
    "Dataset to serve individual images to our model"
    
    def __init__(self, data, transforms=None):
        self.data = data
        self.len = len(data)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img, label = self.data[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
    
    def __len__(self):
        return self.len