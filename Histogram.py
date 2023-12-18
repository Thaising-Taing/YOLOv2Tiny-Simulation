import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data

Output_Layer8_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_Python/Output_Layer8_FWD")
print(Output_Layer8_Sim.shape)
Output_Layer8_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_FPGA/Layer_8_Forward")
# Output_Layer8_FPGA = np.array(Output_Layer8_FPGA)
# Output_Layer8_FPGA = torch.tensor(np.concatenate(Output_Layer8_FPGA, axis=0)).reshape(8, 3, 416, 416)
print(Output_Layer8_FPGA.shape)

# Output_Layer8_List = []
# for i in range(8):
#     images = load_pickle(f"/home/msis/Desktop/Python/yolov2/Output_FPGA1/Image{i}_2nd_result")
#     Output_Layer8_List.append(images)

# Output_Layer8_FPGA = np.array(Output_Layer8_List)
# Output_Layer8_FPGA = torch.tensor(np.concatenate(Output_Layer8_FPGA, axis=0)).reshape(8, 16, 208, 208)
# print(Output_Layer8_FPGA.shape)


# # Reshape the tensor for plotting histograms
data1 = Output_Layer8_Sim # Convert to a 1D NumPy array
data2 = Output_Layer8_FPGA   # Convert to a 1D NumPy array

# _d1 = Output_Layer8_Sim
# _d2 = Output_Layer8_FPGA
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
plt.title('Histograms of Simulation and FPGA - Output_Layer8')
plt.legend()


# Increase figure size and resolution
plt.figure(figsize=(12, 8), dpi=120)
_img = 0
_data1 = data1[_img].view(-1).detach().cpu().numpy()
_data2 = data2[_img].view(-1).detach().cpu().numpy()
plt.hist(_data1, bins=50, alpha=0.5, label=f'Image- Simulation')
plt.hist(_data2, bins=50, alpha=0.5, label=f'Image- FPGA')
plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA - Output_Layer8')
plt.legend()



plt.show()