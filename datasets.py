import torch
from torch.nn.functional import one_hot

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
    
    
class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        n = 7
        drop = 3
        
        arr = torch.arange(n)

        if drop:
            arr = torch.cat([arr[:drop-1], arr[drop:]])
    
        self.data = one_hot(arr, n)
        self.target = one_hot((arr+1)%n, n)
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return len(self.data)