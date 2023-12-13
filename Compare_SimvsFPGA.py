import pickle
import torch 
from tabulate import tabulate
import numpy as np 
import matplotlib.pyplot as plt

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data


if __name__=="__main__":

    # Input Image: Sim vs. FPGA
    Input_Image_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_PyTorch/Input_Image")
    print(Input_Image_Sim.shape)
    Input_Image_FPGA_List = []
    for i in range(8):
        images = load_pickle(f"/home/msis/Desktop/Python/yolov2/Output_FPGA/Image{i}")
        Input_Image_FPGA_List.append(images)
    
    # Check the shapes of images in Input_Image_FPGA_List
    # for i, img in enumerate(Input_Image_FPGA_List):
    #     print(f"Image {i} shape:", img.shape)

    Input_Image_FPGA = np.array(Input_Image_FPGA_List)
    Input_Image_FPGA = torch.tensor(np.concatenate(Input_Image_FPGA, axis=0))

    if torch.equal(Input_Image_Sim, Input_Image_FPGA):
        print("Tensors are identical")
    else:
        print("Tensors are different") # it is different

    # Reshape the tensor for plotting histograms
    data1 = Input_Image_Sim.view(-1).detach().cpu().numpy() # Convert to a 1D NumPy array
    data2 = Input_Image_FPGA.view(-1).detach().cpu().numpy()   # Convert to a 1D NumPy array

    # Increase figure size and resolution
    plt.figure(figsize=(12, 8), dpi=120)

    plt.hist(data1, bins=50, alpha=0.5, label='Simulation Loss_Gradient')
    plt.hist(data2, bins=50, alpha=0.5, label='FPGA Loss_Gradient')

    plt.xlabel('Values')
    plt.ylabel('Count')
    plt.title('Histograms of Simulation and FPGA')
    plt.legend()

    plt.show()
        
    # Input_Image_Min_Value = torch.min(Input_Image_Sim-Input_Image_FPGA)
    # Input_Image_Max_Value = torch.max(Input_Image_Sim-Input_Image_FPGA)
    # Input_Image_Mean_Value = torch.mean(Input_Image_Sim-Input_Image_FPGA)

    # # Output_Image_1st_Iter: Sim vs. FPGA
    # Output_Image_Layer0_1st_Iter_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_Simulation/Output_Layer0_1st_Iter.pickle")
    # Output_Image_Layer0_1st_Iter_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_FPGA/Output_Layer0_1st_Iter.pickle")
    # Output_Image_Layer0_1st_Iter_Min_Value = torch.min(Output_Image_Layer0_1st_Iter_Sim-Output_Image_Layer0_1st_Iter_FPGA)
    # Output_Image_Layer0_1st_Iter_Max_Value = torch.max(Output_Image_Layer0_1st_Iter_Sim-Output_Image_Layer0_1st_Iter_FPGA)
    # Output_Image_Layer0_1st_Iter_Mean_Value = torch.mean(Output_Image_Layer0_1st_Iter_Sim-Output_Image_Layer0_1st_Iter_FPGA)

    # # Output_Image_2nd_Iter: Sim vs. FPGA
    # Output_Image_Layer0_2nd_Iter_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_Simulation/Output_Layer0_2nd_Iter.pickle")
    # Output_Image_Layer0_2nd_Iter_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_FPGA/Output_Layer0_2nd_Iter.pickle")
    # Output_Image_Layer0_2nd_Iter_Min_Value = torch.min(Output_Image_Layer0_2nd_Iter_Sim-Output_Image_Layer0_2nd_Iter_FPGA)
    # Output_Image_Layer0_2nd_Iter_Max_Value = torch.max(Output_Image_Layer0_2nd_Iter_Sim-Output_Image_Layer0_2nd_Iter_FPGA)
    # Output_Image_Layer0_2nd_Iter_Mean_Value = torch.mean(Output_Image_Layer0_2nd_Iter_Sim-Output_Image_Layer0_2nd_Iter_FPGA)

    # # Output_Image__Layer8_1st_Iter_Layer8: Sim vs. FPGA
    # Output_Image_Layer8_1st_Iter_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_Simulation/Output_Layer8_2nd_Iter.pickle")
    # Output_Image_Layer8_1st_Iter_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_FPGA/Output_Layer8_2nd_Iter.pickle")
    # Output_Image_Layer8_1st_Iter_Min_Value = torch.min(Output_Image_Layer8_1st_Iter_Sim-Output_Image_Layer8_1st_Iter_FPGA)
    # Output_Image_Layer8_1st_Iter_Max_Value = torch.max(Output_Image_Layer8_1st_Iter_Sim-Output_Image_Layer8_1st_Iter_FPGA)
    # Output_Image_Layer8_1st_Iter_Mean_Value = torch.mean(Output_Image_Layer8_1st_Iter_Sim-Output_Image_Layer8_1st_Iter_FPGA)

    # # Weight_Layer0: Sim vs. FPGA
    # Weight_Layer0_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_Simulation/Weight_Layer0.pickle")
    # Weight_Layer0_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_FPGA/Weight_Layer0.pickle")
    # Weight_Layer0_Min_Value = torch.min(Weight_Layer0_Sim-Weight_Layer0_FPGA)
    # Weight_Layer0_Max_Value = torch.max(Weight_Layer0_Sim-Weight_Layer0_FPGA)
    # Weight_Layer0_Mean_Value = torch.mean(Weight_Layer0_Sim-Weight_Layer0_FPGA)

    # # Weight_Layer8: Sim vs. FPGA
    # Weight_Layer8_Sim = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_Simulation/Weight_Layer8.pickle")
    # Weight_Layer8_FPGA = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Pickle_FPGA/Weight_Layer8.pickle")
    # Weight_Layer8_Min_Value = torch.min(Weight_Layer8_Sim-Weight_Layer8_FPGA)
    # Weight_Layer8_Max_Value = torch.max(Weight_Layer8_Sim-Weight_Layer8_FPGA)
    # Weight_Layer8_Mean_Value = torch.mean(Weight_Layer8_Sim-Weight_Layer8_FPGA)

    # data = [
    #     ["Input_Image_Value", Input_Image_Min_Value, Input_Image_Max_Value, Input_Image_Mean_Value],
    #     ["Output_Image_Layer0_1st_Iter_Value", Output_Image_Layer0_1st_Iter_Min_Value, Output_Image_Layer0_1st_Iter_Max_Value, Output_Image_Layer0_1st_Iter_Mean_Value],
    #     ["Output_Image_Layer0_2nd_Iter_Value", Output_Image_Layer0_2nd_Iter_Min_Value, Output_Image_Layer0_2nd_Iter_Max_Value, Output_Image_Layer0_2nd_Iter_Mean_Value],
    #     ["Output_Image_Layer8_Value", Output_Image_Layer8_1st_Iter_Min_Value, Output_Image_Layer8_1st_Iter_Max_Value, Output_Image_Layer8_1st_Iter_Mean_Value],
    #     ["Weight_Layer0_Value", Weight_Layer0_Min_Value, Weight_Layer0_Max_Value, Weight_Layer0_Mean_Value],
    #     ["Weight_Layer8_Value", Weight_Layer8_Min_Value, Weight_Layer8_Max_Value, Weight_Layer8_Mean_Value]
    # ]

    # # Display the table using tabulate
    # table = tabulate(data, headers=['Variable (Sim Vs. FPGA)', 'Min Value', 'Max Value', 'Mean Value'], tablefmt='grid')
    # print(table)

