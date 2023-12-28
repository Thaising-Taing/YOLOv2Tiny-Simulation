import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data

Input_Grad_Layer0_PyTorch = load_pickle("/home/msis/Desktop/Python/yolov2/Thaising_Sim_PyTorch/Input_Grad_Layer2")
print(Input_Grad_Layer0_PyTorch[0][0][0][0:20])
Input_Grad_Layer0_Python = load_pickle("/home/msis/Desktop/Python/yolov2/Thaising_Sim_Python/Input_Grad_Layer2")
# Input_Grad_Layer0_FPGA = np.array(Input_Grad_Layer0_FPGA)
# Input_Grad_Layer0_FPGA = torch.tensor(np.concatenate(Input_Grad_Layer0_FPGA, axis=0)).reshape(8, 3, 416, 416)
print(Input_Grad_Layer0_Python[0][0][0][0:20])

# Input_Grad_Layer0_List = []
# for i in range(8):
#     images = load_pickle(f"/home/msis/Desktop/Python/yolov2/Output_FPGA1/Image{i}_2nd_result")
#     Input_Grad_Layer0_List.append(images)

# Input_Grad_Layer0_FPGA = np.array(Input_Grad_Layer0_List)
# Input_Grad_Layer0_FPGA = torch.tensor(np.concatenate(Input_Grad_Layer0_FPGA, axis=0)).reshape(8, 16, 208, 208)
# print(Input_Grad_Layer0_FPGA.shape)


# # Reshape the tensor for plotting histograms
data1 = Input_Grad_Layer0_PyTorch # Convert to a 1D NumPy array
data2 = Input_Grad_Layer0_Python   # Convert to a 1D NumPy array

# _d1 = Input_Grad_Layer0_Sim
# _d2 = Input_Grad_Layer0_FPGA
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
    # axs[i].hist(_data1, bins=50, alpha=0.5, label=f'Image {i} - Simulation')
    # axs[i].hist(_data2, bins=50, alpha=0.5, label=f'Image {i} - FPGA')
    axs[i].hist(_data1, bins=50, alpha=0.5, label=f'Image-Simulation')
    axs[i].hist(_data2, bins=50, alpha=0.5, label=f'Image-FPGA')
    axs[i].set_xlim([-0.005, 0.005])
plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA - Input_Grad_Layer0')
plt.legend()


# Increase figure size and resolution
plt.figure(figsize=(12, 8), dpi=120)
_img = 2
_data1 = data1[_img].view(-1).detach().cpu().numpy()
_data2 = data2[_img].view(-1).detach().cpu().numpy()
plt.hist(_data1, bins=50, alpha=0.5, label=f'Image- Simulation')
plt.hist(_data2, bins=50, alpha=0.5, label=f'Image- FPGA')
plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA - Input_Grad_Layer0')
plt.legend()



plt.show()