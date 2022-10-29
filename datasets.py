import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import random

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


class TranslateDataset(Dataset):
    def __init__(self, diff=1, drops = None) -> None:
        n = 5
        if drops is None:
            drops = [random.randint(1, n-1)]
        # drops = [3]
        
        arr = torch.arange(n)
        

        # for drop in drops:
        #     arr = torch.cat([arr[:drop-1], arr[drop:]])

        test = random.randint(1, n-1)
        # arr = torch.tensor([test, test])
        
        data = one_hot(arr, n)
        targets = one_hot((arr+diff)%n, n)
        
        
        # data =  torch.concat((data, data), 0)
        # targets =  torch.concat((targets, targets), 0)
        
        
        
        super().__init__(data, targets)


# class TranslateDataset2(Dataset):
#     def __init__(self) -> None:
#         arr = torch.arange(n)
#         if drop:
#             arr = torch.cat([arr[:drop-1], arr[drop:]])
    
#         super().__init__(one_hot(arr, n), one_hot((arr-1)%n, n))
        

class TimeDataset1(Dataset):
    def __init__(self):
        self.data = torch.tensor([[0,1], [0,1], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[1,0,0,0]])
     
class TimeDataset2(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,0], [1,0], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[0,1,0,0]])
        
    
class TimeDataset3(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,0], [0,1], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[0,0,1,0]])
        
class TimeDataset4(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,1], [0,1], [0,0]])
        self.target = torch.tensor([[0,0,0,0], [0,0,0,0],[0,0,0,1]])
        
     
                            
     
class MetaDataset():
    """
    Dataset of datasets
    """
    def __init__(self):
        # self.datasets = [dataset() for dataset in [TranslateDataset, TranslateDataset2]]
        # self.datasets = [dataset() for dataset in [TranslateDataset]]
        # self.datasets = [dataset() for dataset in [TimeDataset1, TimeDataset2, TimeDataset3, TimeDataset4]]
        # self.datasets = [dataset() for dataset in [FTDataset2, FTDataset3]]
        self.init()
        
        
    def init(self):
        self.datasets = [
            TranslateDataset(1),
            TranslateDataset(0),
            TranslateDataset(2),
            # TranslateDataset(3),
            TranslateDataset(-1),
            # TranslateDataset(-2),
            ]
        
    def iterate(self):
        """
        returns a generator that gives an shuffled index
        """
        self.init()
        return iter(torch.randperm(len(self.datasets)))

    @property
    def n_inputs(self):
        return self.datasets[0].data.shape[1]

    @property
    def n_outputs(self):
        return self.datasets[0].target.shape[1]
    
    def get_set(self, n):
        return DataLoader(self.datasets[n] , batch_size=1, shuffle=True)
    
    def get_set_size(self):
        return len(self.datasets[0])