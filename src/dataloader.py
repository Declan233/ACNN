import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, trs, label, trace_num):
        self.trs = trs
        self.label = label
        self.trace_num = trace_num

    def __getitem__(self, i):
        index = i % self.trace_num
        trace = torch.from_numpy(self.trs[index]).unsqueeze(0)  # Change shape to (1, trace_size)
        if self.label is not None:
            label = torch.from_numpy(self.label[index])
            return trace.float(), label
        else:
            return trace.float()

    def __len__(self):
        return self.trace_num
    
    
def _create_data_loader(batch_size, kwargs, shuffle=True, drop_last=False):
    """Creates a DataLoader for training."""
    dataset = TorchDataset(**kwargs)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            drop_last=drop_last, 
                            num_workers=1, 
                            pin_memory=True)
    return dataloader
