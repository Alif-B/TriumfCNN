from torch.utils.data import DataLoader, RandomSampler
from dynamic_sampler import DynamicSampler
from h5_dataset import H5Dataset
from CNN_mpmt_datadet import CNNmPMTDataset

filepath = "NonPyFiles/event998.h5"
position_file = "NonPyFiles/mpmt_image_positions.npz"

h5 = CNNmPMTDataset(filepath, position_file)
sampler = RandomSampler(h5)

loader = DataLoader(h5, batch_size=5, sampler=sampler)

iter = iter(loader)
first = next(iter)

print(first)






# To iterate over the dataset
# for i in loader.dataset:
#     print(i["event_ids"], end=' ')


# One way to iterate over dataloader I found online (TypeError)
# dataiter = iter(loader)
# data = dataiter.next()
# features, label = data
# print(features)
# print(label)

