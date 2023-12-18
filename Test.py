import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data

Weight_Grad_Layer8 = load_pickle("/home/msis/Desktop/Python/yolov2/Output_Sim_PyTorch/Weight_Gradient_Layer8")
# print(Weight_Grad_Layer8.shape)
with open("Weight_Gradient_Layer8.txt", mode="w") as Wr: 
    Wr.write(str(Weight_Grad_Layer8))