import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader


class SelfOrganizeTest(torch.nn.Module):
    """
    If Rule can learn this data, then it can somewhat self organize
    """
    def __init__(self) :
        super().__init__()
        self.data = torch.tensor([[0,1], [1,0]])
        self.target = torch.tensor([[0,1], [1,0]])
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)

class ANDDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,0], [0,1],[1,0], [1,1]])
        self.target = torch.tensor([[1,0],[1,0],[1,0],[0,1]])
        
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)
    
    
class XORDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,0], [0,1],[1,0], [1,1]])
        self.target = torch.tensor([[1,0],[0,1],[0,1],[1,0]])
        
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)
    
    
class Dataset(torch.utils.data.Dataset):
    """
    datset superclass
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)    

n = 4
drop = 3

class TranslateDataset(Dataset):
    def __init__(self) -> None:
        
        arr = torch.arange(n)

        if drop:
            arr = torch.cat([arr[:drop-1], arr[drop:]])
    
        super().__init__(one_hot(arr, n), one_hot((arr+1)%n, n))


class TranslateDataset2(Dataset):
    def __init__(self) -> None:
        arr = torch.arange(n)
        if drop:
            arr = torch.cat([arr[:drop-1], arr[drop:]])
    
        super().__init__(one_hot(arr, n), one_hot((arr-1)%n, n))
        
    
class MetaDataset():
    """
    Dataset of datasets
    """
    def __init__(self):
        self.datasets = [dataset() for dataset in [TranslateDataset, TranslateDataset2]]
        
    def iterate(self):
        """
        returns a generator that gives an index
        """
        yield from range(len(self.datasets))
        
    def get_set(self, n):
        return DataLoader(self.datasets[n] , batch_size=1, shuffle=True)