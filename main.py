import torch
from torch import optim
import pdb
from copy import deepcopy
import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from pypcie import Device
from ast import literal_eval
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import subprocess
import tqdm
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import os
import sys
sys.path.append("../")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"Dataset"))
sys.path.append(os.path.join(os.getcwd(),"src"))
sys.path.append(os.path.join(os.getcwd(),"src/GiTae"))
sys.path.append(os.path.join(os.getcwd(),"src/config"))
sys.path.append(os.path.join(os.getcwd(),"src/Main_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Pre_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Post_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Weight_Update_Algorithm"))
sys.path.append(os.path.join(os.getcwd(),"src/Wathna"))
sys.path.append("/home/msis/Desktop/pcie_python/GUI")
# from  XdmaAccess import XdmaAccess
from Pre_Processing_Scratch.Pre_Processing import *
from Pre_Processing_Scratch.Pre_Processing_Function import *
import time
import os.path 
from Dataset.roidb import RoiDataset, detection_collate
from Dataset.factory import get_imdb
from torch.utils.data import DataLoader
import pickle
from Post_Processing_Scratch.Post_Processing_2Iterations_Training_Inference import *
from Detection.Detection import *
from Weight_Update_Algorithm.weight_update import *
from Weight_Update_Algorithm.yolov2_tiny import *
from Weight_Update_Algorithm.Shoaib import Shoaib_Code
from Weight_Update_Algorithm.yolov2tiny_LightNorm_2Iterations import Yolov2

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
from Wathna_pytorch import Pytorch
from Wathna_python import Python
from Thaising_PyTorch import TorchSimulation
from Thaising_Python import PythonSimulation
# from Junaid import Junaid
from GiTae import FPGA

DDR_SIZE = 0x10000
MAX_LINE_LENGTH = 1000

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

save_debug_data = True


class App(customtkinter.CTk):
    
    def __init__(self):
        super().__init__()
        
                
        self.DDR_SIZE = 0x10000
        self.MAX_LINE_LENGTH = 1000

        # Floating Point Parameters that we use
        self.FloatingPoint_Format = "FP32"

        # For the BFP16, we don't need to select format, we just use Single Precision, Truncated and Rounding

        self.Selected_FP_Format = {
            "FP32": (8, 23),
            "Bfloat16": (5, 10),
            "Custom": (7, 8)
        }

        self.Exponent_Bits, self.Mantissa_Bits = self.Selected_FP_Format.get(self.FloatingPoint_Format, (0, 0))

        # Pre-define Conditions:
        self.Mode = "Training"  # Declare whether is Training or Inference
        # Mode = "Inference"  # Declare whether is Training or Inference

        # Pre-Processing Pre-Defined Conditions
        self.Brain_Floating_Point    = True  # Declare Whether Bfloat16 Conversion or not
        # Signals for Debugging Purposes
        self.Image_Converted         = True  # Converted Images
        self.Weight_Converted        = True  # Converted Weight
        self.Bias_Converted          = True  # Converted Bias
        self.BN_Param_Converted      = True  # Converted BN Parameters
        self.Microcode               = True  # Microcode Writing TODO: Check Microcode
        # TODO: Check Microcode

        # Running Hardware Model:
        self.YOLOv2_Hardware_Forward     = True
        self.YOLOv2_Hardware_Backward    = True

        self.phase_forward = 'Forward'
        self.phase_backward = 'Backward'

        # configure window
        self.title("Yolov2 Accelerator.py")
        self.geometry(f"{165+145+10+600-120+10+170+10+170+10}x{855-20}")
        
        
        #customtkinter.set_appearance_mode("Dark")
        #top_frame
        self.top_frame = customtkinter.CTkFrame(self, width=1140, height=96)
        self.top_frame.place(x=20, y=20)
        
        try: original_image = Image.open("1.png")
        except: original_image = Image.open("src/GiTae/1.png")
        resized_image = original_image.resize((177, 96), Image.ANTIALIAS)  # Adjust the size as needed
        self.image = ImageTk.PhotoImage(resized_image)

        background_color = (255, 255, 255)  # White background color
        non_transparent_image = Image.new("RGB", resized_image.size, background_color)
        non_transparent_image.paste(resized_image, (0, 0), resized_image)
        
        self.image = ImageTk.PhotoImage(non_transparent_image)

        self.top_image_label = customtkinter.CTkLabel(self.top_frame, image=self.image, text="")
        self.top_image_label.place(x=0, y=0)
        top_label = customtkinter.CTkLabel(self.top_frame, text="YOLOv2 Training Accelerator", font=("Helvetica", 60))
        top_label.place(x=190, y=15)
                     
        

        background_color = (255, 255, 255)  # White background color
        
        
        
        # Cover
        self.cover = customtkinter.CTkTextbox(self, width=165+145+10+600-120+10+170+10+170+10, height=680, bg_color="#EBEBEB", fg_color="#EBEBEB")
        self.cover.place(x=0, y=130)
        
        
        
        
       
        # Model Selection Frame
        self.mode_frame = customtkinter.CTkFrame(self, width=145, height=680)  # 너비 값 조정
        self.mode_frame.place(x=10, y=140)
        
        self.mode_label = customtkinter.CTkLabel(self.mode_frame, text="Mode Selection", font=("Helvetica", 15))
        self.mode_label.place(x=15, y=10)  
        
        # Mode Selection Frame
        button_width = 120
        button_height = 30

        self.PythonMode = customtkinter.CTkButton(self.mode_frame, text="Python"      , command=self.PythonMode_click , width=button_width, height=button_height)
        self.PythonMode.place(x=10, y=50)
        
        self.PyTorchMode = customtkinter.CTkButton(self.mode_frame, text="PyTorch"    , command=self.PyTorchMode_click , width=button_width, height=button_height)
        self.PyTorchMode.place(x=10, y=100)
        
        self.PythonSim = customtkinter.CTkButton(self.mode_frame, text="Python Sim"   , command=self.PythonSim_click , width=button_width, height=button_height)
        self.PythonSim.place(x=10, y=150)

        self.PytorchSim = customtkinter.CTkButton(self.mode_frame, text="Pytorch Sim" , command=self.PytorchSim_click , width=button_width, height=button_height)
        self.PytorchSim.place(x=10, y=200)
        
        self.PythonCUDA = customtkinter.CTkButton(self.mode_frame, text="Python CUDA" , command=self.PythonCUDA_click , width=button_width, height=button_height)
        self.PythonCUDA.place(x=10, y=250)

        self.FPGA_Python = customtkinter.CTkButton(self.mode_frame, text="FPGA"       , command=self.FPGA_Python_click , width=button_width, height=button_height)
        self.FPGA_Python.place(x=10, y=300)

        self.ResetMode = customtkinter.CTkButton(self.mode_frame, text="Reset Mode"   , command=self.Reset_Mode_click , width=button_width, height=button_height)
        self.ResetMode.place(x=10, y=350)


       
       
       
        # Button Frame 
        self.left_frame = customtkinter.CTkFrame(self, width=145, height=680)  # 너비 값 조정
        self.left_frame.place(x=165, y=140)
        
        
        self.left_label = customtkinter.CTkLabel(self.left_frame, text="Control Panel", font=("Helvetica", 15))
        self.left_label.place(x=10, y=10)  
        
        #left_frame_button
        button_width = 120
        button_height = 30
        self.Load_PCIe  = customtkinter.CTkButton(  self.left_frame, 
                                                    text="Load PCIe",
                                                    command=self.Load_PCIe_click, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Load_PCIe.place(x=10, y=50)

        self.Load_Data = customtkinter.CTkButton(   self.left_frame, 
                                                    text="Load Data",
                                                    command=self.Load_Data_click, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Load_Data.place(x=10, y=100)
        
        self.Load_Microcode_Train = customtkinter.CTkButton(self.left_frame, 
                                                    text="Microcode Train",
                                                    command=self.Load_Microcode_Train_click, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Load_Microcode_Train.place(x=10, y=150)
        
        self.Load_Microcode_Infer = customtkinter.CTkButton(self.left_frame, 
                                                    text="Microcode Infer",
                                                    command=self.Load_Microcode_Infer_click, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Load_Microcode_Infer.place(x=10, y=200)

        self.Train= customtkinter.CTkButton(        self.left_frame, 
                                                    text="Train ",
                                                    command=self.Run_Train, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Train .place(x=10, y=250)
        
        self.Infer = customtkinter.CTkButton(       self.left_frame, 
                                                    text="Infer",
                                                    command=self.Run_Infer, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Infer.place(x=10, y=300)

        self.Stop = customtkinter.CTkButton(        self.left_frame, 
                                                    text="Stop",
                                                    command=self.Stop_Process, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Stop.place(x=10, y=350)
        
        # self.Reset = customtkinter.CTkButton(        self.left_frame, 
        #                                             text="Reset",
        #                                             command=self.Reset_process, 
        #                                             width=button_width, 
        #                                             height=button_height,
        #                                             state='disabled'
        #                                             )
        # self.Reset.place(x=10, y=350)

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=600-120, height=680)  # Adjust the height value as needed
        self.textbox.place(x=165+145+10, y=140)

        # right_frame_1
        self.right_frame_1 = customtkinter.CTkFrame(self, width=170, height=680)
        self.right_frame_1.place(x=165+145+10+600-120+10, y=140)
        if True:
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer Status", font=("Helvetica", 15))
            text_label.place(x=40, y=5)
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer1  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=45)
            self.L1_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L1_FW_canvas.place(x=120, y=45)
            self.L1_FW = self.L1_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer2  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=80)
            self.L2_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L2_FW_canvas.place(x=120, y=80)
            self.L2_FW = self.L2_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer3  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=115)
            self.L3_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L3_FW_canvas.place(x=120, y=115)
            self.L3_FW = self.L3_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer4  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=150)
            self.L4_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L4_FW_canvas.place(x=120, y=150)
            self.L4_FW = self.L4_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer5  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=185)
            self.L5_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L5_FW_canvas.place(x=120, y=185)
            self.L5_FW = self.L5_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer6  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=220)
            self.L6_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L6_FW_canvas.place(x=120, y=220)
            self.L6_FW = self.L6_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer7  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=255)
            self.L7_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L7_FW_canvas.place(x=120, y=255)
            self.L7_FW = self.L7_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer8  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=290)
            self.L8_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L8_FW_canvas.place(x=120, y=290)
            self.L8_FW = self.L8_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer9  FW", font=("Helvetica", 15))
            text_label.place(x=20, y=325)
            self.L9_FW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L9_FW_canvas.place(x=120, y=325)
            self.L9_FW = self.L9_FW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer9  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=360)
            self.L9_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L9_BW_canvas.place(x=120, y=360)
            self.L9_BW = self.L9_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer8  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=395)
            self.L8_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L8_BW_canvas.place(x=120, y=395)
            self.L8_BW = self.L8_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer7  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=430)
            self.L7_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L7_BW_canvas.place(x=120, y=430)
            self.L7_BW = self.L7_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer6  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=465)
            self.L6_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L6_BW_canvas.place(x=120, y=465)
            self.L6_BW = self.L6_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer5  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=500)
            self.L5_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L5_BW_canvas.place(x=120, y=500)
            self.L5_BW = self.L5_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer4  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=535)
            self.L4_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L4_BW_canvas.place(x=120, y=535)
            self.L4_BW = self.L4_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer3  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=570)
            self.L3_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L3_BW_canvas.place(x=120, y=570)
            self.L3_BW = self.L3_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer2  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=605)
            self.L2_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L2_BW_canvas.place(x=120, y=605)
            self.L2_BW = self.L2_BW_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_1, text="Layer1  BW", font=("Helvetica", 15))
            text_label.place(x=20, y=640)
            self.L1_BW_canvas = customtkinter.CTkCanvas(self.right_frame_1, width=23, height=23)
            self.L1_BW_canvas.place(x=120, y=640)
            self.L1_BW = self.L1_BW_canvas.create_oval(3, 3, 23, 23, fill="red")


        #right_frame_2
        self.right_frame_2 = customtkinter.CTkFrame(self, width=170, height=680)
        self.right_frame_2.place(x=165+145+10+600-120+10+170+10, y=140)
        if True:
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="IRQ Status", font=("Helvetica", 15))
            text_label.place(x=50, y=5)
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer1  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=60)
            self.L1_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L1_IRQ_canvas.place(x=120, y=60)
            self.L1_IRQ = self.L1_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer2  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=120)
            self.L2_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L2_IRQ_canvas.place(x=120, y=120)
            self.L2_IRQ = self.L2_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer3  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=180)
            self.L3_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L3_IRQ_canvas.place(x=120, y=180)
            self.L3_IRQ = self.L3_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer4  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=240)
            self.L4_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L4_IRQ_canvas.place(x=120, y=240)
            self.L4_IRQ = self.L4_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer5  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=300)
            self.L5_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L5_IRQ_canvas.place(x=120, y=300)
            self.L5_IRQ = self.L5_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer6  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=360)
            self.L6_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L6_IRQ_canvas.place(x=120, y=360)
            self.L6_IRQ = self.L6_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer7  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=420)
            self.L7_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L7_IRQ_canvas.place(x=120, y=420)
            self.L7_IRQ = self.L7_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer8  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=480)
            self.L8_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L8_IRQ_canvas.place(x=120, y=480)
            self.L8_IRQ = self.L8_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")
            
            text_label = customtkinter.CTkLabel(self.right_frame_2, text="Layer9  IRQ", font=("Helvetica", 15))
            text_label.place(x=20, y=540)
            self.L9_IRQ_canvas = customtkinter.CTkCanvas(self.right_frame_2, width=23, height=23)
            self.L9_IRQ_canvas.place(x=120, y=540)
            self.L9_IRQ = self.L9_IRQ_canvas.create_oval(3, 3, 23, 23, fill="red")           

        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        
        self.Show_Text("GUI Ready.")


    # Helper Functions
    def change_color(self, canvas, item, color):
        canvas.itemconfig(item, fill=color)  
        self.update()

    def change_color_red(self):
        self.change_color(self.L1_IRQ_canvas, self.L1_IRQ, "red")
        self.change_color(self.L2_IRQ_canvas, self.L2_IRQ, "red")
        self.change_color(self.L3_IRQ_canvas, self.L3_IRQ, "red")
        self.change_color(self.L4_IRQ_canvas, self.L4_IRQ, "red")
        self.change_color(self.L5_IRQ_canvas, self.L5_IRQ, "red")
        self.change_color(self.L6_IRQ_canvas, self.L6_IRQ, "red")
        self.change_color(self.L7_IRQ_canvas, self.L7_IRQ, "red")
        self.change_color(self.L8_IRQ_canvas, self.L8_IRQ, "red")
        self.change_color(self.L9_IRQ_canvas, self.L9_IRQ, "red")    
        self.update()  

    def Show_Text(self, text, end=[]):
        if not end==[]:
            self.textbox.insert("0.0", text + "\n")
        else:
            self.textbox.insert("0.0", text + "\n\n")
        self.textbox.update_idletasks()
        self.textbox.see("end")
        print(text)

    def Clear_Text(self):
        self.textbox.delete("0.0", tkinter.END)    
        self.textbox.update_idletasks()


    # Select Modes
    def PyTorchMode_click(self):
        self.mode =  "Pytorch"
        self.PyTorchMode.configure(state="disabled")
        self.PyTorchMode.configure(fg_color='green')
        self.PythonMode.configure(state="disabled")
        self.PytorchSim.configure(state="disabled")
        self.PythonSim.configure(state="disabled")
        self.PythonCUDA.configure(state="disabled")
        self.FPGA_Python.configure(state="disabled")
        self.ResetMode.configure(state="normal")
        
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        self.update()
        self.Show_Text(f"Pytorch mode selected.")
        self.update()
        
    def PythonMode_click(self):
        self.mode =  'Python'
        self.PyTorchMode.configure(state="disabled")
        self.PythonMode.configure(state="disabled")
        self.PythonMode.configure(fg_color='green')
        self.PytorchSim.configure(state="disabled")
        self.PythonSim.configure(state="disabled")
        self.PythonCUDA.configure(state="disabled")
        self.FPGA_Python.configure(state="disabled")
        self.ResetMode.configure(state="normal")
        
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        self.Show_Text(f"Python mode selected.")
        self.update()
        
    def PytorchSim_click(self):
        self.mode =  'PytorchSim'
        self.PyTorchMode.configure(state="disabled")
        self.PythonMode.configure(state="disabled")
        self.PytorchSim.configure(state="disabled")
        self.PytorchSim.configure(fg_color='green')
        self.PythonSim.configure(state="disabled")
        self.PythonCUDA.configure(state="disabled")
        self.FPGA_Python.configure(state="disabled")
        self.ResetMode.configure(state="normal")
        
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
       
        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        self.Show_Text(f"Simulation mode selected.")
        self.update()
             
    def PythonSim_click(self):
        self.mode =  'PythonSim'
        self.PyTorchMode.configure(state="disabled")
        self.PythonMode.configure(state="disabled")
        self.PytorchSim.configure(state="disabled")
        self.PythonSim.configure(state="disabled")
        self.PythonSim.configure(fg_color='green')
        self.PythonCUDA.configure(state="disabled")
        self.FPGA_Python.configure(state="disabled")
        self.ResetMode.configure(state="normal")
        
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        self.Show_Text(f"Simulation mode selected.")
        self.update()
        
    def PythonCUDA_click(self):
        self.mode =  'PythonCUDA'
        self.PyTorchMode.configure(state="disabled")
        self.PythonMode.configure(state="disabled")
        self.PytorchSim.configure(state="disabled")
        self.PythonSim.configure(state="disabled")
        self.PythonCUDA.configure(state="disabled")
        self.PythonCUDA.configure(fg_color='green')
        self.FPGA_Python.configure(state="disabled")
        self.ResetMode.configure(state="normal")
        
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lift(self.cover)
        self.right_frame_2.lift(self.cover)
        self.Show_Text(f"FPGA mode selected.")
        self.update()
    
    def FPGA_Python_click(self):
        self.mode =  'FPGA'
        self.PyTorchMode.configure(state="disabled")
        self.PythonMode.configure(state="disabled")
        self.PytorchSim.configure(state="disabled")
        self.PythonSim.configure(state="disabled")
        self.PythonCUDA.configure(state="disabled")
        self.FPGA_Python.configure(state="disabled")
        self.FPGA_Python.configure(fg_color='green')
        self.ResetMode.configure(state="normal")
        
        self.Load_PCIe.configure(state="normal")
        self.Load_Data.configure(state="normal")
        self.Load_Microcode_Train.configure(state="normal")
        self.Load_Microcode_Infer.configure(state="normal")
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lift(self.cover)
        self.right_frame_2.lift(self.cover)
        self.Show_Text(f"FPGA mode selected.")
        self.update()
        
    def Reset_Mode_click(self):
        self.mode =  4
        self.PyTorchMode.configure(state="enabled")
        self.PythonMode.configure(state="enabled")
        self.PytorchSim.configure(state="enabled")
        self.PythonSim.configure(state="enabled")
        self.PythonCUDA.configure(state="enabled")
        self.FPGA_Python.configure(state="enabled")
        self.ResetMode.configure(state="normal")
        
        self.PyTorchMode.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.PythonMode.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.PytorchSim.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.PythonSim.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.PythonCUDA.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.FPGA_Python.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.Load_PCIe.configure(state="disabled")
        self.Load_Data.configure(state="disabled")
        self.Load_Microcode_Train.configure(state="disabled")
        self.Load_Microcode_Infer.configure(state="disabled")
        self.Train.configure(state="disabled")
        self.Infer.configure(state="disabled")
        self.Stop.configure(state="disabled")
        
        self.Load_PCIe.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Load_Data.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Load_Microcode_Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Load_Microcode_Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        self.Clear_Text()
        self.Show_Text(f"Reset.")
        self.update()
            
    # Select Operations        
    def Load_PCIe_click(self):
        self.Show_Text(f"Load PCIe Driver.")
        exit_code = subprocess.call(['sudo', "./load_driver.sh"], cwd='./src/t_cbu/test2')
    
    def Load_Data_click(self):
        self.Show_Text(f"Load Data to FPGA.")

        self.d = Device("0000:08:00.0")
        # Access BAR 0
        self.bar = self.d.bar[2]
        # self.bar = XdmaAccess(0)
        i = 0x0
        j = 0x0
        k = 0x0
        l = 0x0 
        # data_to_write = []
        # data_to_write.clear()
        # file_path = "Weight_CH0.txt"
        # #with open("32bit_flip_line/YOLOv2_Weight_32_flip_8.txt") as file:
        # try:
        #     with open(file_path, "r", encoding="utf-8") as file:
        #         for line in file:
        #             # 16진수 문자열을 10진수 정수로 변환
        #             line_data = int(line.strip(), 16)
        #             data_to_write.append(line_data)
        # except FileNotFoundError:
        #     print(f"파일을 찾을 수 없습니다: {file_path}")
        # except ValueError as e:
        #     print(f"데이터를 정수로 변환하는 중 오류 발생: {str(e)}")
        # except Exception as e:
        #     print(f"파일을 읽어오는 중 오류 발생: {str(e)}")   
        # data_to_write_array = np.array(data_to_write, dtype=np.uint32)
        # self.bar.write_dma(0x80000000, data_to_write_array)    

        with open("Weight_CH0.txt") as file:
          for item in file:
            #print(item) 
            an_integer = int(item, 16)
           # hex_value = hex(an_integer)
            self.bar.write(0x0000 +i, an_integer)
            #print(an_integer)
            i = i + 0x4

        print("Weight0 Done!")

        # data_to_write = []
        # data_to_write.clear()

        #with open("32bit_flip_line/YOLOv2_Image_32_flip_8.txt") as file:
        # file_path = "Image_CH0.txt"
        # try:
        #     with open(file_path, "r", encoding="utf-8") as file:
        #         for line in file:
        #             # 16진수 문자열을 10진수 정수로 변환
        #             line_data = int(line.strip(), 16)
        #             data_to_write.append(line_data)
        # except FileNotFoundError:
        #     print(f"파일을 찾을 수 없습니다: {file_path}")
        # except ValueError as e:
        #     print(f"데이터를 정수로 변환하는 중 오류 발생: {str(e)}")
        # except Exception as e:
        #     print(f"파일을 읽어오는 중 오류 발생: {str(e)}")   
        # data_to_write_array = np.array(data_to_write, dtype=np.uint32)
        # self.bar.write_dma(0x82400000, data_to_write_array)    


        with open("Image_CH0.txt") as file:
          for item in file:
            #print(item) 
            an_integer = int(item, 16)
           # hex_value = hex(an_integer)
            self.bar.write(0x2400000 +j, an_integer)
            #print(an_integer)
            j = j + 0x4

        print("Image0 Done!")

        # data_to_write = []
        # data_to_write.clear()

        # file_path = "Weight_CH1.txt"
        # try:
        #     with open(file_path, "r", encoding="utf-8") as file:
        #         for line in file:
        #             # 16진수 문자열을 10진수 정수로 변환
        #             line_data = int(line.strip(), 16)
        #             data_to_write.append(line_data)
        # except FileNotFoundError:
        #     print(f"파일을 찾을 수 없습니다: {file_path}")
        # except ValueError as e:
        #     print(f"데이터를 정수로 변환하는 중 오류 발생: {str(e)}")
        # except Exception as e:
        #     print(f"파일을 읽어오는 중 오류 발생: {str(e)}")   
        # data_to_write_array = np.array(data_to_write, dtype=np.uint32)
        # self.bar.write_dma(0x90000000, data_to_write_array)    

        with open("Weight_CH1.txt") as file:
          for item in file:
            #print(item) 
            an_integer = int(item, 16)
           # hex_value = hex(an_integer)
            #bar.write(0x8000000 +k, an_integer)
            self.bar.write(0x10000000 +k, an_integer) # 512MB
            #print(an_integer)
            k = k + 0x4

        print("Weight1 Done!")

        # data_to_write = []
        # data_to_write.clear()

        # file_path = "Image_CH1.txt"
        # try:
        #     with open(file_path, "r", encoding="utf-8") as file:
        #         for line in file:
        #             # 16진수 문자열을 10진수 정수로 변환
        #             line_data = int(line.strip(), 16)
        #             data_to_write.append(line_data)
        # except FileNotFoundError:
        #     print(f"파일을 찾을 수 없습니다: {file_path}")
        # except ValueError as e:
        #     print(f"데이터를 정수로 변환하는 중 오류 발생: {str(e)}")
        # except Exception as e:
        #     print(f"파일을 읽어오는 중 오류 발생: {str(e)}")   
        # data_to_write_array = np.array(data_to_write, dtype=np.uint32)
        # self.bar.write_dma(0x92400000, data_to_write_array)  

        with open("Image_CH1.txt") as file:
          for item in file:
            #print(item) 
            an_integer = int(item, 16)
           # hex_value = hex(an_integer)
            #bar.write(0x9200000 +l, an_integer)
            self.bar.write(0x12400000 +l, an_integer) #512MB
            #print(an_integer)
            l = l + 0x4

        print("Image1 Done!")

        print("Write Done!")

    def Load_Microcode_Train_click(self):
        self.Show_Text(f"Load Microcode to FPGA.")
        
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]
        #self.textbox.insert("0.0", "CTkTextbox\n\n" )

        # microcode = Microcode("src/GiTae/Forward.txt") 
        microcode = Microcode("src/GiTae/MICROCODE.txt")
        

        for i in range (0, len(microcode)):
            self.bar.write(0x4, microcode[i]) # wr mic
            self.bar.write(0x8, i) # wr addr
            self.bar.write(0x0, 0x00000012) # wr en
            self.bar.write(0x0, 0x00000010) # wr en low
        print("mic write done")    
        
    def Load_Microcode_Infer_click(self):
        self.Show_Text(f"Load Microcode to FPGA.")
        
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]
        #self.textbox.insert("0.0", "CTkTextbox\n\n" )

        microcode = Microcode("src/GiTae/Forward.txt") 
        # microcode = Microcode("src/GiTae/MICROCODE.txt")
        

        for i in range (0, len(microcode)):
            self.bar.write(0x4, microcode[i]) # wr mic
            self.bar.write(0x8, i) # wr addr
            self.bar.write(0x0, 0x00000012) # wr en
            self.bar.write(0x0, 0x00000010) # wr en low
        print("mic write done")    
        
    def Run_Train(self):  
        self.Train.configure(state="disabled")
        self.Train.configure(fg_color='green')
        self.Infer.configure(state="disabled")
        self.Stop.configure(state="normal")
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.Show_Text(f"Start Training")
        self.Show_Text(f"Mode : {self.mode}")
        
        self.parse_args()
        self.Pre_Process()
        self.Create_Output_Dir()
        self.Load_Weights()
        self.Load_Dataset()

        for self.epoch in range(self.args.start_epoch, self.args.max_epochs):
            self.whole_process_start = time.time()
            self.data_iter = iter(self.train_dataloader)
            # self.data_iter = iter(self.small_train_dataloader)
            self.Adjust_Learning_Rate()
            
            for step in tqdm(range(self.iters_per_epoch_train), desc=f"Training for Epoch {self.epoch}", total=self.iters_per_epoch_train):
            # for step in tqdm(range(self.iters_per_epoch_train_subset), desc=f"Training for Epoch {self.epoch}", total=self.iters_per_epoch_train_subset):
                self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.data_iter)
                # if save_debug_data: self.Save_File(next(self.data_iter), "Dataset/Dataset/default_data.pickle")
                self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = self.Load_File("Dataset/Dataset/default_data.pickle")
                self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = self.Load_File("Dataset/Dataset/default_data0.pickle")
                # self.show_image(self.im_data[0])
                
                self.Before_Forward() ######################### - Individual Functions
                self.Forward() ################################ - Individual Functions
                # self.Visualize()
                self.Calculate_Loss()
                self.Before_Backward() ######################## - Individual Functions
                self.Backward() ############################### - Individual Functions
                self.Weight_Update() 
                
            if self.epoch%4 == 0: self.Check_mAP()
            self.save_weights()
        #     self.Save_Pickle()
        self.Post_Epoch()
        self.Show_Text(f"Training is finished")

    def Run_Infer(self):
        self.Train.configure(state="disabled")
        self.Infer.configure(state="disabled")
        self.Infer.configure(fg_color='green')
        self.Stop.configure(state="normal")
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Show_Text(f"Start Inference")
        self.parse_args()
        self.Pre_Process()
        self.Create_Output_Dir()
        self.Load_Weights()
        self.Load_Dataset()

        self.whole_process_start = time.time()
        self.data_iter = iter(self.test_dataloader)
        
        for step in tqdm(range(self.iters_per_epoch_test), desc=f"Inference", total=self.iters_per_epoch_test):
            # self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.data_iter)
            if save_debug_data: self.Save_File(next(self.data_iter), "Dataset/Dataset/default_data.pickle")
            # self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = self.Load_File("Dataset/Dataset/default_data.pickle")
            
            self.batch = step
            self.Before_Forward() ######################### - Individual Functions
            self.Forward_Infer() ################################ - Individual Functions
            self.Visualize()
            # self.Visualize_All()
              
        self.Show_Text(f"Total Images with detections   : {self.count['detections']}")
        self.Show_Text(f"Total Images without detections: {self.count['no_detections']}")
        self.Show_Text(f"Inference is finished.")
        
    def Run_Validation(self):
        self.Train.configure(state="disabled")
        self.Infer.configure(state="disabled")
        self.Infer.configure(fg_color='green')
        self.Stop.configure(state="normal")
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Show_Text(f"Start validation")
        self.parse_args()
        self.Pre_Process()
        self.Create_Output_Dir()
        self.Load_Weights()
        self.Load_Dataset()
        
        # For validation
        self.img_id = -1
        self.val_imdb = get_imdb("voc_2007_test")
        self.all_boxes = [[[] for _ in range(len(self.val_imdb.image_index))] for _ in range(self.val_imdb.num_classes)]

        self.whole_process_start = time.time()
        self.data_iter = iter(self.small_test_dataloader)
        for step in tqdm(range(self.iters_per_epoch_test), desc=f"Validation", total=self.iters_per_epoch_test):
            self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.data_iter)
            # if save_debug_data: self.Save_File(next(self.data_iter), "Dataset/Dataset/default_data.pickle")
            # self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = self.Load_File("Dataset/Dataset/default_data0.pickle")
            
            self.batch = step
            self.Before_Forward() ######################### - Individual Functions
            self.Forward() ################################ - Individual Functions
            self.Validate()
        
        
        with open(os.path.join(self.args.output_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # map = val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)
        map = self.val_imdb.evaluate_detections_with_train(self.all_boxes, output_dir=self.args.output_dir)
        # return map   
        
        # self.Show_Text(f"Total Images with detections   : {self.count['detections']}")
        # self.Show_Text(f"Total Images without detections: {self.count['no_detections']}")
        # self.Show_Text(f"Inference is finished.")
        
    def Stop_Process(self):
        self.Train.configure(state="normal")
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(state="normal")
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(state="disabled")
        self.Stop.configure(fg_color='green')
        self.Show_Text(f"Stop the process")
        
        
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : 
            self.d = Device("0000:08:00.0")
            self.bar = self.d.bar[0]
            self.bar.write(0x0, 0x00000011) # yolo start
            self.bar.write(0x0, 0x00000010) # yolo start low

            self.bar.write(0x8, 0x00000011) # rd addr
            self.bar.write(0x0, 0x00000014) # rd en
            self.bar.write(0x0, 0x00000010) # rd en low

            self.bar.write(0x18, 0x00008001) # axi addr
            self.bar.write(0x14, 0x00000001) # axi rd en
            self.bar.write(0x14, 0x00000000) # axi rd en low
        
        
        
        
        
    # Training Helper Functions
    def Save_File(self, path, data):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def Load_File(self, path):
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return b
    
    def parse_args(self):
        """
        Parse input arguments
        """
        parser = argparse.ArgumentParser(description='Yolo v2')
        parser.add_argument('--max_epochs', dest='max_epochs',
                            help='number of epochs to train',
                            default=1, type=int)
        parser.add_argument('--start_epoch', dest='start_epoch',
                            default=0, type=int)
        parser.add_argument('--total_training_set', dest='total_training_set',
                            default=16000, type=int)
        parser.add_argument('--total_inference_set', dest='total_inference_set',
                            default=619, type=int)
        parser.add_argument('--batch_size', dest='batch_size',
                            default=8, type=int)
        parser.add_argument('--nw', dest='num_workers',
                            help='number of workers to load training data',
                            default=2, type=int)
        parser.add_argument('--use_small_dataset', dest='use_small_dataset',
                            default=True, type=bool)
        parser.add_argument('--save_interval', dest='save_interval',
                            default=10, type=int)
        parser.add_argument('--pretrained', dest='pretrained',
                            # default="Dataset/Dataset/pretrained/yolov2_best_map.pth", type=str)
                            default="Dataset/Dataset/pretrained/yolov2_epoch_100_2iteration.pth", type=str)
        parser.add_argument('--output_dir', dest='output_dir',
                            default="Output", type=str)
        parser.add_argument('--cuda', dest='use_cuda',
                            default=True, type=bool)
        parser.add_argument('--vis', dest='vis',
                            default=False, type=bool)

        self.args = parser.parse_args()
         
    def get_dataset(self,datasetnames):
        names = datasetnames.split('+')
        dataset = RoiDataset(get_imdb(names[0]))
        # print('load dataset {}'.format(names[0]))
        for name in names[1:]:
            tmp = RoiDataset(get_imdb(name))
            dataset += tmp
            # print('load and add dataset {}'.format(name))
        return dataset
    
    def Pre_Process(self):
        # To keep a record of weights with best mAP  
        self.max_map, self.best_map_score, self.best_map_epoch, self.best_map_loss = 0, -1, -1, -1
        
        self.classes  = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        
        self.count = dict()
        self.count['detections'], self.count['no_detections'] = 0, 0
                
        if self.mode == "Pytorch"    :  self.Pytorch          = Pytorch(self)
        if self.mode == "Python"     :  self.Python           = Python(self)
        if self.mode == "PythonSim"  :  self.PythonSimulation = PythonSimulation(self)
        if self.mode == "PytorchSim" :  self.TorchSimulation  = TorchSimulation(self)
        if self.mode == "PythonCUDA" :  self.Junaid           = Junaid(self)
        if self.mode == "FPGA"       :  self.FPGA             = FPGA(self)

    def Create_Output_Dir(self):
        if self.mode == "Pytorch"    :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if self.mode == "Python"     :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if self.mode == "PythonSim"  :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if self.mode == "PytorchSim" :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if self.mode == "PythonCUDA" :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if self.mode == "FPGA"       :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

    def Load_Weights(self):
        s = time.time()
        self.PreProcessing = Pre_Processing(Mode =   self.Mode,
                                            Brain_Floating_Point =   self.Brain_Floating_Point,
                                            Exponent_Bits        =   self.Exponent_Bits,
                                            Mantissa_Bits        =   self.Mantissa_Bits)
        self.Weight, self.Bias, self.Gamma, self.Beta, self.Running_Mean_Dec, self.Running_Var_Dec = self.PreProcessing.WeightLoader()
        
        # Initialize Pre-Trained Weight
        self.Shoaib = Shoaib_Code(  Weight_Dec=self.Weight, 
                                    Bias_Dec=self.Bias, 
                                    Beta_Dec=self.Beta, 
                                    Gamma_Dec=self.Gamma,
                                    Running_Mean_Dec=self.Running_Mean_Dec, 
                                    Running_Var_Dec=self.Running_Var_Dec,
                                    args=self.args,
                                    pth_weights_path=self.args.pretrained,
                                    model=Yolov2,
                                    optim=optim)
        
        # Loading Weight From pth File
        self.loaded_weights = self.Shoaib.load_weights()
        Weight, Bias, Gamma_WeightBN, BetaBN, Running_Mean_Dec, Running_Var_Dec = self.loaded_weights
                
        if self.mode == "Pytorch"    :  self.Pytorch.load_weights(self.loaded_weights)
        if self.mode == "Python"     :  self.Python.load_weights(self.loaded_weights)
        if self.mode == "PythonSim"  :  self.PythonSimulation.load_weights(self.loaded_weights)
        if self.mode == "PytorchSim" :  self.TorchSimulation.load_weights(self.loaded_weights)
        if self.mode == "PythonCUDA" :  self.Junaid.load_weights(self.loaded_weights)
        if self.mode == "FPGA"       :  self.FPGA.load_weights(self.loaded_weights)
        
        e = time.time()
        print("WeightLoader : ",e-s)

    def update_weights(self, data):
        [ self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, self.Running_Mean_Dec, self.Running_Var_Dec ] = data

    def Load_Dataset(self):
        self.imdb_train_name = 'voc_2007_trainval+voc_2012_trainval'
        self.train_dataset = self.get_dataset(self.imdb_train_name)
        # Whole Training Dataset 
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=True)
        self.iters_per_epoch_train = int(len(self.train_dataset) / self.args.batch_size)
        # Small Training Dataset
        self.small_train_dataset = torch.utils.data.Subset(self.train_dataset, range(0, self.args.total_training_set))
        self.small_train_dataloader = DataLoader(self.small_train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=True)
        self.iters_per_epoch_train_subset = int(len(self.small_train_dataset) / self.args.batch_size)
        # -------------------------------------- Test Dataset -----------------------------------------------------
        self.imdb_test_name = 'voc_2007_test'
        self.test_dataset = self.get_dataset(self.imdb_test_name)
        # Whole Training Dataset 
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=True)
            # # Small Training Dataset
            # self.small_test_dataset = torch.utils.data.Subset(self.test_dataloader, range(0, self.args.total_inference_set))
            # # print("Sub Training Dataset: " + str(len(small_dataset)))
            # self.s = time.time()
            # self.small_test_dataloader = DataLoader(self.small_test_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=True)
            # self.e = time.time()
            # print("Data Loader : ",self.e-self.s)
        self.iters_per_epoch_test  = int(len(self.test_dataset) / self.args.batch_size)
        
    def Adjust_Learning_Rate(self):
        # learning_rate = 0.001
        self.learning_rate = 0.001
        # Various of Learning will Change with the Epochs    
        if self.epoch >= 10 and self.epoch < 20:
            self.learning_rate = 0.0001
        elif self.epoch >= 20 and self.epoch < 30:
            self.learning_rate = 0.000001
        elif self.epoch >= 30:
            self.learning_rate = 0.0000001
            
    def Before_Forward(self):
        if self.mode == "Pytorch"    :  pass
        if self.mode == "Python"     :  pass
        if self.mode == "PythonSim"  :  pass
        if self.mode == "PytorchSim" :  pass
        if self.mode == "PythonCUDA" :  pass
        if self.mode == "FPGA"       :  self.FPGA.Before_Forward(self)

    def Forward(self):
        if self.mode == "Pytorch"    :  self.Pytorch.Forward(self)
        if self.mode == "Python"     :  self.Python.Forward(self)
        if self.mode == "PythonSim"  :  self.PythonSimulation.Forward(self)
        if self.mode == "PytorchSim" :  self.TorchSimulation.Forward(self)
        if self.mode == "PythonCUDA" :  self.Junaid.Forward(self)
        if self.mode == "FPGA"       :  self.FPGA.Forward(self)
    
    def Forward_Infer(self):
        if self.mode == "Pytorch"    :  self.Pytorch.Forward(self)
        if self.mode == "Python"     :  self.Python.Forward(self)
        if self.mode == "PythonSim"  :  self.PythonSimulation.Forward(self)
        if self.mode == "PytorchSim" :  self.TorchSimulation.Forward(self)
        if self.mode == "PythonCUDA" :  self.Junaid.Forward(self)
        if self.mode == "FPGA"       :  self.FPGA.Forward_Inference(self)
    
    def Calculate_Loss(self):
        if self.mode == "Pytorch"    :  self.Pytorch.Calculate_Loss(self)
        if self.mode == "Python"     :  self.Python.Calculate_Loss(self)
        if self.mode == "PythonSim"  :  self.PythonSimulation.Calculate_Loss(self)
        if self.mode == "PytorchSim" :  self.TorchSimulation.Calculate_Loss(self)
        if self.mode == "PythonCUDA" :  self.Junaid.Calculate_Loss(self)
        if self.mode == "FPGA"       :  self.FPGA.Calculate_Loss(self)

    def Before_Backward(self):
        if self.mode == "Pytorch"    :  pass
        if self.mode == "Python"     :  pass
        if self.mode == "PythonSim"  :  pass
        if self.mode == "PytorchSim" :  pass
        if self.mode == "PythonCUDA" :  pass
        if self.mode == "FPGA"       : self.FPGA.Before_Backward(self)
        
    def Backward(self):
        if self.mode == "Pytorch"    :  self.Pytorch.Backward(self)
        if self.mode == "Python"     :  self.Python.Backward(self)
        if self.mode == "PythonSim"  :  self.PythonSimulation.Backward(self)
        if self.mode == "PytorchSim" :  self.TorchSimulation.Backward(self)
        if self.mode == "PythonCUDA" :  self.Junaid.Backward(self)
        if self.mode == "FPGA"       :  self.FPGA.Backward(self)

    def Weight_Update(self):
        if self.mode == "Pytorch"    :  _data =  self.Pytorch
        if self.mode == "Python"     :  _data =  self.Python
        if self.mode == "PythonSim"  :  _data =  self.PythonSimulation
        if self.mode == "PytorchSim" :  _data =  self.TorchSimulation
        if self.mode == "PythonCUDA" :  _data =  self.Junaid
        if self.mode == "FPGA"       :  _data =  self.FPGA
        
        new_weights, self.custom_model = self.Shoaib.update_weights_FPGA(
                                                                Inputs  = [_data.Weight,  _data.Bias,  _data.Gamma,  _data.Beta], 
                                                                gInputs = [_data.gWeight, _data.gBias, _data.gGamma, _data.gBeta ])
        _data.Weight,  _data.Bias,  _data.Gamma,  _data.Beta = new_weights

        if save_debug_data: self.Save_File("./Output_Sim_Python/Weight_Layer0_After",_data.Weight[0])
        if save_debug_data: self.Save_File("./Output_Sim_Python/Beta_Layer0_After",_data.Beta[0])
        if save_debug_data: self.Save_File("./Output_Sim_Python/Gamma_Layer0_After",_data.Gamma[0])
        
        if self.mode == "Pytorch"    :  self.Pytorch.load_weights(new_weights)
        if self.mode == "Python"     :  self.Python.load_weights(new_weights)
        if self.mode == "PythonSim"  :  self.PythonSimulation.load_weights(new_weights)
        if self.mode == "PytorchSim" :  self.TorchSimulation.load_weights(new_weights)
        if self.mode == "PythonCUDA" :  self.Junaid.load_weights(new_weights)
        if self.mode == "FPGA"       :  self.FPGA.load_weights(new_weights)
            
        [self.Weight, self.Bias, self.Gamma, self.Beta] = new_weights

    def Save_Pickle(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": 
            # Check and save the best mAP
            if self.map > self.max_map:
                self.max_map = self.map
                self.best_map_score = round((self.map*100),2)
                self.best_map_epoch = self.epoch
                self.best_map_loss  = round(self.Loss.item(),2)
                self.save_name = os.path.join(self.args.output_dir, 'yolov2_best_map.pth')
                print(f'\n\t--------------------->>Saving best weights at Epoch {self.epoch}, with mAP={round((self.map*100),2)}% and loss={round(self.Loss.item(),2)}\n')
                torch.save({
                    'model': self.custom_model.state_dict(),
                    'epoch': self.epoch,
                    'loss': self.Loss.item(),
                    'map': map
                    }, self.save_name)
            if self.epoch == 0: pass
            print(f"Epoch: {self.epoch}/{self.args.max_epochs} --Loss: {round(self.Loss.item(),2)} --mAP: {self.map} --Best mAP: {self.best_map_score}  at Epoch {self.best_map_epoch}") 
            # return best_map_score, best_map_epoch, best_map_loss
                
        if self.mode == "FPGA"      : 
            # Save Pickle: 
            if self.curr_epoch % self.args.save_interval == 0:
                self._data = self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec, self.curr_epoch
                self.output_file = os.path.join(self.args.output_dir, f'Params_{self.curr_epoch}.pickle')
                with open(self.output_file, 'wb') as handle:
                    pickle.dump(self._data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    def save_weights(self):
        model = self.Shoaib.custom_model
        save_dir = os.path.join(self.args.output_dir, "trained_weights")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        _now = str(datetime.now()).split()
        save_name = os.path.join(save_dir, f'{_now[0]}-{_now[1]}-Epoch_{self.epoch}.pth') 
        torch.save({
            'model': self.Shoaib.custom_model.state_dict(),
            'epoch': self.epoch,
            'lr': self.Shoaib.custom_optimizer.param_groups[0]['lr'],
            }, save_name)
    
    def Check_mAP(self):
        if self.mode == "Pytorch"    :  _data = self.Pytorch
        if self.mode == "Python"     :  _data = self.Python
        if self.mode == "PythonSim"  :  _data = self.PythonSimulation
        if self.mode == "PytorchSim" :  _data = self.TorchSimulation
        if self.mode == "PythonCUDA" :  _data = self.Junaid
        if self.mode == "FPGA"       :  _data = self.FPGA
        
        self.map = self.Shoaib.cal_mAP(Inputs_with_running = \
            [_data.Weight, _data.Bias, _data.Gamma, _data.Beta, _data.Running_Mean_Dec, _data.Running_Var_Dec])

        date_time = str(datetime.now()).replace(" ","---").split(".")[0].replace(":","-")
        with open("mAP.txt", mode="a+") as output_file_1:
            output_file_1.write(f"{date_time}: {self.map} \n")
                
    def Post_Epoch(self): 
        if self.mode == "Pytorch"    :  _data =  self.Pytorch
        if self.mode == "Python"     :  _data =  self.Python
        if self.mode == "PythonSim"  :  _data =  self.PythonSimulation
        if self.mode == "PytorchSim" :  _data =  self.TorchSimulation
        if self.mode == "PythonCUDA" :  _data =  self.Junaid
        if self.mode == "FPGA"       :  _data =  self.FPGA
        
        self.whole_process_end = time.time()
        self.whole_process_time = self.whole_process_end - self.whole_process_start
        self.output_text = f"Epoch: {self.epoch+1}/{self.args.max_epochs}--Loss: {_data.Loss}"
        self.Show_Text(self.output_text)
    
    def Visualize(self):
        if self.mode == "Pytorch"    :  _data = self.Pytorch
        if self.mode == "Python"     :  _data = self.Python
        if self.mode == "PythonSim"  :  _data = self.PythonSimulation
        if self.mode == "PytorchSim" :  _data = self.TorchSimulation
        if self.mode == "PythonCUDA" :  _data = self.Junaid
        if self.mode == "FPGA"       :  _data = self.FPGA
        
        # if self.mode == "Pytorch"   : if save_debug_data: self.Save_File(_data.out, "output_of_forward_Pytorch.pickle"     )
        # if self.mode == "Python"    : if save_debug_data: self.Save_File(_data.out, "output_of_forward_Python.pickle"      )
        # if self.mode == "Simulation": if save_debug_data: self.Save_File(_data.out, "output_of_forward_Simulation.pickle"  )
        # if self.mode == "FPGA"      : if save_debug_data: self.Save_File(_data.out, "output_of_forward_FPGA.pickle"        )
        
        
        out_batch = _data.out
        
        self.Show_Text(f"Infer - {self.mode} - {out_batch.shape}")
        
        for i, (img,out) in enumerate(zip(_data.image,out_batch)):
            _img = img.cpu().detach().numpy().astype(np.uint8)
            _img = np.transpose(_img, (1,2,0))
            
            im_info = dict()
            im_info['height'], im_info['width'], _  = _img.shape
            
            yolo_output = self.reshape_outputs(out)
            yolo_output = [item[0].data for item in yolo_output]
                
            # detections = yolo_eval(yolo_output, im_info, conf_threshold=0.2, nms_threshold=0.5)
            detections = yolo_eval(yolo_output, im_info, conf_threshold=0.45, nms_threshold=0.5)
            
            if len(detections) > 0:
                det_boxes = detections[:, :5].cpu().numpy()
                det_classes = detections[:, -1].long().cpu().numpy()
                
                temp_image_path = 'Output/temp.jpg'
                plt.imsave(temp_image_path, _img)
                img = Image.open(temp_image_path)
                im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=self.classes)
                        
                self.count['detections']+=1
                self.Show_Text(f"{len(detections)} Detections", end='')
                
                # plt.figure('Input Image')
                # plt.imshow(_img)
                plt.figure(f'Output Image')
                plt.imshow(im2show)
                plt.show(block=True)
                
            else:
                self.count['no_detections']+=1
                self.Show_Text(f"No Detections", end='')
                # self.Show_Text(f"Batch {self.batch} - Image {i+1} -- No Detections", end='')

    def Visualize_All(self):
        if self.mode == "Pytorch"    :  _data = self.Pytorch
        if self.mode == "Python"     :  _data = self.Python
        if self.mode == "PythonSim"  :  _data = self.PythonSimulation
        if self.mode == "PytorchSim" :  _data = self.TorchSimulation
        if self.mode == "PythonCUDA" :  _data = self.Junaid
        if self.mode == "FPGA"       :  _data = self.FPGA
        
        out_batch_torch  = self.Load_File("output_of_forward_Torch.pickle")
        
        out_batch_sim    = self.Load_File("output_of_forward_sim.pickle")
        
        out_batch_fpga   = self.Load_File("output_of_forward_FPGA.pickle")
        
        
        for i, (img,outTorch, outSim, outFPGA) in enumerate(zip(_data.image, out_batch_torch, out_batch_sim, out_batch_fpga)):
            _img = img.cpu().detach().numpy().astype(np.uint8)
            _img = np.transpose(_img, (1,2,0))
            
            im_info = dict()
            im_info['height'], im_info['width'], _  = _img.shape
            
            yolo_output_torch = self.reshape_outputs(outTorch)
            yolo_output_torch = [item[0].data for item in yolo_output_torch]
            
            yolo_output_sim = self.reshape_outputs(outSim)
            yolo_output_sim = [item[0].data for item in yolo_output_sim]
            
            yolo_output_fpga = self.reshape_outputs(outFPGA)
            yolo_output_fpga = [item[0].data for item in yolo_output_fpga]
            
            detections_Torch = yolo_eval(yolo_output_torch, im_info, conf_threshold=0.6, nms_threshold=0.4)
            detections_Sim   = yolo_eval(yolo_output_sim, im_info, conf_threshold=0.6, nms_threshold=0.4)
            detections_FPGA  = yolo_eval(yolo_output_fpga, im_info, conf_threshold=0.6, nms_threshold=0.4)
            
            if len(detections_Torch) > 0 or len(detections_Sim) > 0 or len(detections_FPGA) > 0:
                temp_image_path = 'Output/temp.jpg'
                plt.imsave(temp_image_path, _img)
                imgTorch = Image.open(temp_image_path)
                imgSim   = Image.open(temp_image_path)
                imgFPGA  = Image.open(temp_image_path)
                
                self.Show_Text(f"Batch {self.batch} - Image {i+1} -- Showing Detections", end='')
                
                # Create Figure
                # plt.axis('off')
                fig = plt.figure(f'Output Image')

                # Create a subplot
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.axis('off')
                if len(detections_Torch) > 0:
                    det_boxes_Torch = detections_Torch[:, :5].cpu().numpy()
                    det_classes_Torch = detections_Torch[:, -1].long().cpu().numpy()
                    im2show_Torch = draw_detection_boxes(imgTorch, det_boxes_Torch, det_classes_Torch, class_names=self.classes)
                    # Display the images on the subplots
                    ax1.imshow(im2show_Torch, cmap='gray')
                else:
                    ax1.imshow(imgTorch, cmap='gray')
                ax1.set_title('PyTorch')

                # Create a subplot
                ax2 = fig.add_subplot(1, 3, 2)
                if len(detections_Sim) > 0:
                    det_boxes_Sim = detections_Sim[:, :5].cpu().numpy()
                    det_classes_Sim = detections_Sim[:, -1].long().cpu().numpy()
                    im2show_Sim = draw_detection_boxes(imgSim, det_boxes_Sim, det_classes_Sim, class_names=self.classes)
                    # Display the images on the subplots
                    ax2.imshow(im2show_Sim, cmap='gray')
                else:
                    ax2.imshow(imgSim, cmap='gray')
                ax2.set_title('Simulation (PyTorch)')
                ax2.axis('off')
                    
                # Create a subplot
                ax3 = fig.add_subplot(1, 3, 3)
                if len(detections_FPGA) > 0:
                    det_boxes_FPGA = detections_FPGA[:, :5].cpu().numpy()
                    det_classes_FPGA = detections_FPGA[:, -1].long().cpu().numpy()
                    im2show_FPGA = draw_detection_boxes(imgFPGA, det_boxes_FPGA, det_classes_FPGA, class_names=self.classes)
                    # Display the images on the subplots
                    ax3.imshow(im2show_FPGA, cmap='gray')
                else:
                    ax3.imshow(imgFPGA, cmap='gray')
                ax3.set_title('FPGA')
                ax3.axis('off')

                # Adjust the spacing between subplots
                plt.tight_layout()
                
                # Show the figure
                plt.show(block=False)
                while not plt.waitforbuttonpress(2):
                    pass
                
            else:
                self.count['no_detections']+=1
                self.Show_Text(f"Batch {self.batch} - Image {i+1} -- No Detections", end='')

    def reshape_outputs(self, out, gt_boxes=None, gt_classes=None, num_boxes=None):
        
        out = torch.tensor(out, requires_grad=True)
        out = torch.unsqueeze(out , 0)
        
        scores = out
        bsize, _, h, w = out.shape
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)
        
        return delta_pred, conf_pred, class_pred   

    def Validate(self):
        if self.mode == "Pytorch"    :  _data = self.Pytorch
        if self.mode == "Python"     :  _data = self.Python
        if self.mode == "PythonSim"  :  _data = self.PythonSimulation
        if self.mode == "PytorchSim" :  _data = self.TorchSimulation
        if self.mode == "PythonCUDA" :  _data = self.Junaid
        if self.mode == "FPGA"       :  _data = self.FPGA
        
        out_batch = _data.out
        
        self.Show_Text(f"Validate - {self.mode} - {self.batch} - {out_batch.shape}")
        
        for i, (img,out) in enumerate(zip(_data.image,out_batch)):
            self.img_id += 1
            _img = img.cpu().detach().numpy().astype(np.uint8)
            _img = np.transpose(_img, (1,2,0))
            
            im_info = dict()
            im_info['height'], im_info['width'], _  = _img.shape
            
            yolo_output = self.reshape_outputs(out)
            yolo_output = [item[0].data for item in yolo_output]
                
            detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)
            
            if len(detections) > 0:
                for cls in range(len(self.classes)):
                    inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                    if inds.numel() > 0:
                        cls_det = torch.zeros((inds.numel(), 5))
                        cls_det[:, :4] = detections[inds, :4]
                        cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                        self.all_boxes[cls][self.img_id] = cls_det.cpu().numpy()
            
    def show_image(self, img):
        img = img.permute(1,2,0).numpy().astype(np.uint8)
        plt.figure()
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()

