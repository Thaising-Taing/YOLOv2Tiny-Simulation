import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data

Weight_After_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_PyTorch/Output_2nd_Iter_Layer0")
print(Weight_After_Sim.shape)
Weight_After_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_FPGA/Image0_2nd_result")
print(Weight_After_FPGA.shape)


# Reshape the tensor for plotting histograms
data1 = Weight_After_Sim # Convert to a 1D NumPy array
data2 = Weight_After_FPGA   # Convert to a 1D NumPy array

# _d1 = Weight_After_Sim
# _d2 = Weight_After_FPGA
# d1 = (_d1[5].permute(1,2,0)[:,:,-3:].numpy() > np.median(_d1[-1].permute(1,2,0)[:,:,:3].numpy())).astype(float)
# d2 = (_d2[5].permute(1,2,0)[:,:,-3:].numpy() > np.median(_d2[-1].permute(1,2,0)[:,:,:3].numpy())).astype(float)
# d3 = abs(d1-d2)

# plt.figure()
# plt.imshow(d1)
# plt.figure()
# plt.imshow(d2)
# plt.figure()
# plt.imshow(d3)


fig, axs = plt.subplots(1,data1.shape[0])
for i in range(data1.shape[0]):
    _data1, _data2 = data1[i].view(-1).detach().cpu().numpy(), data2[i].view(-1).detach().cpu().numpy()
    axs[i].hist(_data1, bins=50, alpha=0.5, label=f'Image {i} - Simulation')
    axs[i].hist(_data2, bins=50, alpha=0.5, label=f'Image {i} - FPGA')
    axs[i].set_xlim([-0.005, 0.005])
plt.xlabel('Values')
plt.ylabel('Count')
# plt.title('Histograms of Simulation and FPGA - Weight_After')
plt.legend()


# Increase figure size and resolution
plt.figure(figsize=(12, 8), dpi=120)
_img = 0
_data1 = data1[_img].view(-1).detach().cpu().numpy()
_data2 = data2[_img].view(-1).detach().cpu().numpy()
plt.hist(_data1, bins=50, alpha=0.5, label=f'Image {_img} - Simulation')
plt.hist(_data2, bins=50, alpha=0.5, label=f'Image {_img} - FPGA')
plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA - Weight_After')
plt.legend()



plt.show()