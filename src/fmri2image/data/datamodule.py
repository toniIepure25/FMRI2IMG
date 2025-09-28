from torch.utils.data import Dataset, DataLoader
import torch

class FMRITextDataset(Dataset):
    def __init__(self, X, texts):
        self.X, self.texts = X, texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), (torch.tensor(idx, dtype=torch.long), self.texts[idx])

def make_loaders(X, texts, batch_size=2, num_workers=0):
    ds = FMRITextDataset(X, texts)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
