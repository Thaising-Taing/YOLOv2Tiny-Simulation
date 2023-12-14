import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data


_Weight_Gradient_initial = load_pickle("Weight_Gradient/Layer_8_Backward_Weight_Gradient_initial")
_Weight_Gradient_old = [sum(map(float, item)) / len(item) for item in zip(*_Weight_Gradient_initial)]
_Weight_Gradient_new = list(np.mean(np.array(_Weight_Gradient_initial), axis=0) )


Output_1st_Iter_Layer0_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_PyTorch/Output_Last_Layer")
print(Output_1st_Iter_Layer0_Sim.shape)
Output_1st_Iter_Layer0_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_FPGA1/Output_last_layer")
print(Output_1st_Iter_Layer0_FPGA.shape)
# Output_1st_Iter_Layer0_List = []
# for i in range(8):
#     images = load_pickle(f"/home/msis/Desktop/Python/yolov2/Output_FPGA1/Image{i}_2nd_result")
#     Output_1st_Iter_Layer0_List.append(images)

# Output_1st_Iter_Layer0_FPGA = np.array(Output_1st_Iter_Layer0_List)
# Output_1st_Iter_Layer0_FPGA = torch.tensor(np.concatenate(Output_1st_Iter_Layer0_FPGA, axis=0)).reshape(8, 16, 208, 208)
# print(Output_1st_Iter_Layer0_FPGA.shape)

# Reshape the tensor for plotting histograms
data1 = Output_1st_Iter_Layer0_Sim.view(-1).detach().cpu().numpy() # Convert to a 1D NumPy array
data2 = Output_1st_Iter_Layer0_FPGA.view(-1).detach().cpu().numpy()   # Convert to a 1D NumPy array

# Increase figure size and resolution
plt.figure(figsize=(12, 8), dpi=120)

plt.hist(data1, bins=50, alpha=0.5, label='Simulation Output_1st_Iter_Layer0')
plt.hist(data2, bins=50, alpha=0.5, label='FPGA Output_1st_Iter_Layer0')

plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA')
plt.legend()
plt.show()
