import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data

Output_1st_Layer0_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_PyTorch/Output_1st_Iter_Layer0")
Output_1st_Layer0_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_FPGA/Output_1st_iter_layer0")

# Reshape the tensor for plotting histograms
data1 = Output_1st_Layer0_Sim.view(-1).detach().cpu().numpy() # Convert to a 1D NumPy array
data2 = Output_1st_Layer0_FPGA.view(-1).detach().cpu().numpy()   # Convert to a 1D NumPy array

# Increase figure size and resolution
plt.figure(figsize=(12, 8), dpi=120)

plt.hist(data1, bins=50, alpha=0.5, label='Simulation Output_1st_Layer0')
plt.hist(data2, bins=50, alpha=0.5, label='FPGA Output_1st_Layer0')

plt.xlabel('Values')
plt.ylabel('Count')
plt.title('Histograms of Simulation and FPGA')
plt.legend()

plt.show()
