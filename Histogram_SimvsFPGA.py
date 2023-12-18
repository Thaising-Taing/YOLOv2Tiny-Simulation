import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data


# _Weight_Gradi0nAfterial = load_pickle("Weight_Gradi0nAfterr_8_Backward_Weight_Gradi0nAfterial")
# _Weight_Gradi0nAfter= [sum(map(float, item)) / len(item) for item in zip(*_Weight_Gradi0nAfterial)]
# _Weight_Gradi0nAfter= list(np.mean(np.array(_Weight_Gradi0nAfterial), axis=0) )


Weight_Layer0_Before_Layer0_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_PyTorch/Weight_Layer0_Before")
print(Weight_Layer0_Before_Layer0_Sim.shape)
Weight_Layer0_Before_Layer0_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_FPGA/Layer_0_Forward_weight_Before_Weight_Update")
print(Weight_Layer0_Before_Layer0_FPGA.shape)
# Weight_Layer0_Before_Layer0_List = []
# for i in range(8):
#     images = load_pickle(f"/home/msis/Desktop/Python/yolov2/Output_FPGA1/Image{i}_2nd_result")
#     Weight_Layer0_Before_Layer0_List.append(images)

# Weight_Layer0_Before_Layer0_FPGA = np.array(Weight_Layer0_Before_Layer0_List)
# Weight_Layer0_Before_Layer0_FPGA = torch.tensor(np.concatenate(Weight_Layer0_Before_Layer0_FPGA, axis=0)).reshape(8, 16, 208, 208)
# print(Weight_Layer0_Before_Layer0_FPGA.shape)

# Reshape the tensor for plotting histograms
data1 = Weight_Layer0_Before_Layer0_Sim.view(-1).detach().cpu().numpy() # Convert to a 1D NumPy array
data2 = Weight_Layer0_Before_Layer0_FPGA.view(-1).detach().cpu().numpy()   # Convert to a 1D NumPy array

# Increase figure size and resolution
plt.figure(figsize=(12, 8), dpi=120)

plt.hist(data1, bins=50, alpha=0.5, label='Simulation Weight_Layer0_Before_Layer0')
plt.hist(data2, bins=50, alpha=0.5, label='FPGA Weight_Layer0_Before_Layer0')

plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA')
plt.legend()
plt.show()
