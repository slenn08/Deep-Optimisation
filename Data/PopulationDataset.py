from torch.utils.data import DataLoader, Dataset

class PopulationDataset(Dataset):
    def __init__(self, X):
        # Extract solutions from population
        self.X = list(map(lambda x : x[0], X))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return {"solution" : self.X[index]}