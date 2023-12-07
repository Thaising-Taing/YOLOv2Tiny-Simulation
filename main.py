import torch
from torch import optim
import pdb
import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from pypcie import Device
from ast import literal_eval
import subprocess
import tqdm
import warnings
warnings.filterwarnings("ignore")

from tkinter.constants import DISABLED, NORMAL 

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
from Post_Processing_Scratch.Post_Processing_2Iterations import Post_Processing
import time
from tabulate import tabulate
import os.path 
import matplotlib.pyplot as plt
from Dataset.roidb import RoiDataset, detection_collate
from Dataset.factory import get_imdb
from torch.utils.data import DataLoader
import pickle
from Post_Processing_Scratch.Post_Processing_2Iterations_Training_Inference import *
from Detection.Detection import *
from Weight_Update_Algorithm.weight_update import *
from Weight_Update_Algorithm.yolov2_tiny import *
from Pre_Processing_Scratch.Neural_Network_Operations_LightNorm import *
from Weight_Update_Algorithm.Shoaib import Shoaib_Code
from Weight_Update_Algorithm.yolov2tiny_LightNorm_2Iterations import Yolov2
# from src.Wathna.torch_2iteration import *
# from src.Wathna.python_original import *

from GiTae_Functions import *

DDR_SIZE = 0x10000


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"



class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
                
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
        self.Microcode               = True  # Microcode Writing

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
        self.Mode1 = customtkinter.CTkButton(self.mode_frame, text="PyTorch"    , command=self.Mode1_click , width=button_width, height=button_height)
        self.Mode1.place(x=10, y=50)

        self.Mode2 = customtkinter.CTkButton(self.mode_frame, text="Python"     , command=self.Mode2_click , width=button_width, height=button_height)
        self.Mode2.place(x=10, y=100)
        
        self.Mode3 = customtkinter.CTkButton(self.mode_frame, text="Simulation" , command=self.Mode3_click , width=button_width, height=button_height)
        self.Mode3.place(x=10, y=150)

        self.Mode4 = customtkinter.CTkButton(self.mode_frame, text="FPGA"       , command=self.Mode4_click , width=button_width, height=button_height)
        self.Mode4.place(x=10, y=200)

        self.ResetMode = customtkinter.CTkButton(self.mode_frame, text="Reset Mode", command=self.Reset_Mode , width=button_width, height=button_height)
        self.ResetMode.place(x=10, y=250)


       
       
       
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
        
        self.Load_Microcode = customtkinter.CTkButton(self.left_frame, 
                                                    text="Load Microcode",
                                                    command=self.Load_Microcode_click, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Load_Microcode.place(x=10, y=150)

        self.Train= customtkinter.CTkButton(        self.left_frame, 
                                                    text="Train ",
                                                    command=self.Run_Train, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Train .place(x=10, y=200)
        
        self.Infer = customtkinter.CTkButton(       self.left_frame, 
                                                    text="Infer",
                                                    command=self.Run_Infer, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Infer.place(x=10, y=250)

        self.Stop = customtkinter.CTkButton(        self.left_frame, 
                                                    text="Stop",
                                                    command=self.Stop_Process, 
                                                    width=button_width, 
                                                    height=button_height,
                                                    state='disabled'
                                                    )
        self.Stop.place(x=10, y=300)
        
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

    def Show_Text(self, text):
        self.textbox.insert("0.0", text + "\n\n")
        self.textbox.update_idletasks()
        self.textbox.see("end")

    def Clear_Text(self):
        self.textbox.delete("0.0", tkinter.END)    
        self.textbox.update_idletasks()


    # Select Modes
    def Mode1_click(self):
        self.mode =  "Pytorch"
        self.Mode1.configure(state="disabled")
        self.Mode1.configure(fg_color='green')
        self.Mode2.configure(state="disabled")
        self.Mode3.configure(state="disabled")
        self.Mode4.configure(state="disabled")
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
        print(f"Pytorch mode selected.")
        self.Show_Text(f"Pytorch mode selected.")
        self.update()
        
    def Mode2_click(self):
        self.mode =  'Python'
        self.Mode1.configure(state="disabled")
        self.Mode2.configure(state="disabled")
        self.Mode2.configure(fg_color='green')
        self.Mode3.configure(state="disabled")
        self.Mode4.configure(state="disabled")
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
        print(f"Python mode selected.")
        self.Show_Text(f"Python mode selected.")
        self.update()
        
    def Mode3_click(self):
        self.mode =  'Simulation'
        self.Mode1.configure(state="disabled")
        self.Mode2.configure(state="disabled")
        self.Mode3.configure(state="disabled")
        self.Mode3.configure(fg_color='green')
        self.Mode4.configure(state="disabled")
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
        print(f"Simulation mode selected.")
        self.Show_Text(f"Simulation mode selected.")
        self.update()
        
    def Mode4_click(self):
        self.mode =  'FPGA'
        self.Mode1.configure(state="disabled")
        self.Mode2.configure(state="disabled")
        self.Mode3.configure(state="disabled")
        self.Mode4.configure(state="disabled")
        self.Mode4.configure(fg_color='green')
        self.ResetMode.configure(state="normal")
        
        self.Load_PCIe.configure(state="normal")
        self.Load_Data.configure(state="normal")
        self.Load_Microcode.configure(state="normal")
        self.Train.configure(state="normal")
        self.Infer.configure(state="normal")
        self.Stop.configure(state="normal")
        
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lift(self.cover)
        self.right_frame_2.lift(self.cover)
        print(f"FPGA mode selected.")
        self.Show_Text(f"FPGA mode selected.")
        self.update()
        
    def Reset_Mode(self):
        self.mode =  4
        self.Mode1.configure(state="normal")
        self.Mode2.configure(state="normal")
        self.Mode3.configure(state="normal")
        self.Mode4.configure(state="normal")
        self.ResetMode.configure(state="disabled")
        
        self.Mode1.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Mode2.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Mode3.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Mode4.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.Load_PCIe.configure(state="disabled")
        self.Load_Data.configure(state="disabled")
        self.Load_Microcode.configure(state="disabled")
        self.Train.configure(state="disabled")
        self.Infer.configure(state="disabled")
        self.Stop.configure(state="disabled")
        
        self.Load_PCIe.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Load_Data.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Load_Microcode.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        
        self.cover.lower()
        self.right_frame_1.lower(self.cover)
        self.right_frame_2.lower(self.cover)
        self.Clear_Text()
        print(f"Reset.")
        self.Show_Text(f"Reset.")
        self.update()
      
      
    # Select Operations        
    def Load_PCIe_click(self):
        print("Load PCIe Driver!")
        self.Show_Text(f"Load PCIe Driver.")
        exit_code = subprocess.call(['sudo', "./load_driver.sh"], cwd='./src/t_cbu/test2')
    
    def Load_Data_click(self):
        print("Load Data to FPGA")
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

    def Load_Microcode_click(self):
        print("Load Microcode to FPGA")
        self.Show_Text(f"Load Microcode to FPGA.")
        
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]
        #self.textbox.insert("0.0", "CTkTextbox\n\n" )

        microcode = Microcode("src/GiTae/Forward.txt") 
        #microcode = Microcode("src/GiTae/MICROCODE.txt")
        

        for i in range (0, len(microcode)):
            self.bar.write(0x4, microcode[i]) # wr mic
            self.bar.write(0x8, i) # wr addr
            self.bar.write(0x0, 0x00000012) # wr en
            self.bar.write(0x0, 0x00000010) # wr en low
        print("mic write done")    
        
    def Run_Train(self):  
        print(f"Start Training")
        self.Show_Text(f"Start Training")
        print(f'Mode : {self.mode}')
        
        self.parse_args()
        self.Pre_Process()
        self.Create_Output_Dir()
        self.Load_Weights()
        self.Load_Dataset()

        for self.epoch in range(self.args.start_epoch, self.args.max_epochs):
            self.whole_process_start = time.time()
            self.train_data_iter = iter(self.small_dataloader)
            self.Adjust_Learning_Rate()
            
            for step in tqdm(range(self.iters_per_epoch), desc=f"Training for Epoch {self.epoch}", total=self.iters_per_epoch):
                self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.train_data_iter)
                self.Before_Forward()
                self.Forward()
                self.Calculate_Loss()
                self.Before_Backward()
                self.Backward()
                self.Weight_Update() 
            self.Check_mAP()
            self.Save_Pickle()
        self.Post_Epoch()

    def Run_Infer(self):
        self.Train.configure(state="disabled")
        self.Infer.configure(state="disabled")
        self.Infer.configure(fg_color='green')
        self.Stop.configure(state="normal")
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        print(f"Start Inference")

        self.Show_Text(f"Start Inference")
        
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass      

        
    def Stop_Process(self):
        self.Train.configure(state="normal")
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(state="normal")
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(state="disabled")
        self.Stop.configure(fg_color='green')
        print("Stop the process")
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
    def parse_args(self):
        """
        Parse input arguments
        """
        parser = argparse.ArgumentParser(description='Yolo v2')
        parser.add_argument('--max_epochs', dest='max_epochs',
                            help='number of epochs to train',
                            default=100, type=int)
        parser.add_argument('--start_epoch', dest='start_epoch',
                            default=0, type=int)
        parser.add_argument('--total_training_set', dest='total_training_set',
                            default=16, type=int)
        parser.add_argument('--total_inference_set', dest='total_inference_set',
                            default=10, type=int)
        parser.add_argument('--batch_size', dest='batch_size',
                            default=8, type=int)
        parser.add_argument('--nw', dest='num_workers',
                            help='number of workers to load training data',
                            default=2, type=int)
        parser.add_argument('--use_small_dataset', dest='use_small_dataset',
                            default=True, type=bool)
        parser.add_argument('--save_interval', dest='save_interval',
                            default=10, type=int)
        parser.add_argument('--output_dir', dest='output_dir',
                            default="Output", type=str)
        parser.add_argument('--cuda', dest='use_cuda',
                            default=True, type=bool)

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
        
        if self.mode=="Pytorch": 
            self.PreProcessing = Pre_Processing(Mode =   self.Mode,
                                Brain_Floating_Point =   self.Brain_Floating_Point,
                                Exponent_Bits        =   self.Exponent_Bits,
                                Mantissa_Bits        =   self.Mantissa_Bits)
        elif self.mode=="Python":
            self.PreProcessing = Pre_Processing(Mode =   self.Mode,
                                Brain_Floating_Point =   self.Brain_Floating_Point,
                                Exponent_Bits        =   self.Exponent_Bits,
                                Mantissa_Bits        =   self.Mantissa_Bits)
        elif self.mode=="Simulation":
            self.PreProcessing = Pre_Processing(Mode =   self.Mode,
                                Brain_Floating_Point =   self.Brain_Floating_Point,
                                Exponent_Bits        =   self.Exponent_Bits,
                                Mantissa_Bits        =   self.Mantissa_Bits)
        elif self.mode=="FPGA":
            self.PreProcessing = Pre_Processing(Mode =   self.Mode,
                                Brain_Floating_Point =   self.Brain_Floating_Point,
                                Exponent_Bits        =   self.Exponent_Bits,
                                Mantissa_Bits        =   self.Mantissa_Bits)

    def Create_Output_Dir(self, out_path = "Output"):
        
        if self.mode == "Pytorch"   :
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
        if self.mode == "Python"    :
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
        if self.mode == "Simulation": 
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)
        if self.mode == "FPGA"      : pass
        
         # Create Directory
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    def Load_Weights(self):
        if self.mode == "Pytorch"   :
            for i in range(8):
                grads[f'gamma{i}'] = grads[f'gamma{i}'].view(-1)
            for i in range(8):
                grads[f'beta{i}'] = grads[f'beta{i}'].view(-1)
                
            update_weights_modular(pytorch_model.params, grads, custom_model, custom_optimizer, pytorch_model)
                    
            # Replace values in Pytorch_model (such as hardware)
            update_model(custom_model, pytorch_model)
            
        if self.mode == "Python"    : 
            for i in range(8): grads[f'gamma{i}'] = grads[f'gamma{i}'].view(-1)
            for i in range(8): grads[f'beta{i}'] = grads[f'beta{i}'].view(-1)
                
            update_weights_modular(pytorch_model.params, grads, custom_model, custom_optimizer, pytorch_model)
                    
            # Replace values in Pytorch_model (such as hardware)
            update_model(custom_model, pytorch_model)
        if self.mode == "Simulation": 
            args = parse_args()
            
            self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec = self.PreProcessing.WeightLoader()
            
            # Initialize Pre-Trained Weight
            self.Shoaib = Shoaib_Code(
                Weight_Dec=self.Weight_Dec, 
                Bias_Dec=self.Bias_Dec, 
                Beta_Dec=self.Beta_Dec, 
                Gamma_Dec=self.Gamma_Dec,
                Running_Mean_Dec=self.Running_Mean_Dec, 
                Running_Var_Dec=self.Running_Var_Dec,
                args=self.args,
                pth_weights_path="/data/Circuit_Team/Thaising/yolov2/src/Pre_Processing_Scratch/data/pretrained/yolov2_best_map.pth",
                model=Yolov2,
                optim=optim)

            # Loading Weight From pth File: 
            self.update_weights(self.Shoaib.load_weights())
            
        if self.mode == "FPGA"      : 
            # Code by GiTae 
            s = time.time()
            self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec = self.PreProcessing.WeightLoader()
            
            # Initialize Pre-Trained Weight: Adding By Thaising
            Shoaib = Shoaib_Code(
                Weight_Dec=self.Weight_Dec, 
                Bias_Dec=self.Bias_Dec, 
                Beta_Dec=self.Beta_Dec, 
                Gamma_Dec=self.Gamma_Dec,
                Running_Mean_Dec=self.Running_Mean_Dec, 
                Running_Var_Dec=self.Running_Var_Dec,
                args=self.args,
                pth_weights_path="/home/msis/training/yolov2/src/Pre_Processing_Scratch/data/pretrained/yolov2_best_map.pth",
                model=Yolov2,
                optim=optim)
            
            # Loading Weight From pth File: Adding By Thaising
            self.update_weights(Shoaib.load_weights())
            
            e = time.time()
            print("WeightLoader : ",e-s)

    def update_weights(self, data):
        [ self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, self.Running_Mean_Dec, self.Running_Var_Dec ] = data

    def Load_Dataset(self):
        if self.mode == "Pytorch"   : 
            args = parse_args()
            self.imdb_name = 'voc_2007_trainval+voc_2012_trainval'
            self.train_dataset = self.get_dataset(self.imdb_name)
            print("Training Dataset: " + str(len(self.train_dataset)))
            # Whole Training Dataset 
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=False)
            # Small Training Dataset
            self.small_dataset = torch.utils.data.Subset(self.train_dataset, range(0, self.args.total_training_set))
            print("Sub Training Dataset: " + str(len(self.small_dataset)))
            self.s = time.time()
            self.small_dataloader = DataLoader(self.small_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=False)
            self.e = time.time()
            print("Data Loader : ",self.e-self.s)
            self.iters_per_epoch = int(len(self.small_dataset) / self.args.batch_size)
        if self.mode == "Python"    : 
            args = parse_args()
            self.imdb_name = 'voc_2007_trainval+voc_2012_trainval'
            self.train_dataset = self.get_dataset(self.imdb_name)
            print("Training Dataset: " + str(len(self.train_dataset)))
            # Whole Training Dataset 
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=False)
            # Small Training Dataset
            self.small_dataset = torch.utils.data.Subset(self.train_dataset, range(0, self.args.total_training_set))
            print("Sub Training Dataset: " + str(len(self.small_dataset)))
            self.s = time.time()
            self.small_dataloader = DataLoader(self.small_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=False)
            self.e = time.time()
            print("Data Loader : ",self.e-self.s)
            self.iters_per_epoch = int(len(self.small_dataset) / self.args.batch_size)
        if self.mode == "Simulation": # Add By Thaising
            args = parse_args()
            self.imdb_name = 'voc_2007_trainval+voc_2012_trainval'
            self.train_dataset = self.get_dataset(self.imdb_name)
            print("Training Dataset: " + str(len(self.train_dataset)))
            # Whole Training Dataset 
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=False)
            # Small Training Dataset
            self.small_dataset = torch.utils.data.Subset(self.train_dataset, range(0, self.args.total_training_set))
            print("Sub Training Dataset: " + str(len(self.small_dataset)))
            self.s = time.time()
            self.small_dataloader = DataLoader(self.small_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=False)
            self.e = time.time()
            print("Data Loader : ",self.e-self.s)
            self.iters_per_epoch = int(len(self.small_dataset) / self.args.batch_size)
        if self.mode == "FPGA"      : 
            # Code by GiTae : 
            self.imdb_name = 'voc_2007_trainval+voc_2012_trainval'
            self.train_dataset = self.get_dataset(self.imdb_name)
            # Whole Training Dataset 
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=True)
            # Small Training Dataset
            self.small_dataset = torch.utils.data.Subset(self.train_dataset, range(0, self.args.total_training_set))
            # print("Sub Training Dataset: " + str(len(small_dataset)))
            self.s = time.time()
            self.small_dataloader = DataLoader(self.small_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=detection_collate, drop_last=True)
            self.e = time.time()
            print("Data Loader : ",self.e-self.s)
            self.iters_per_epoch = int(len(self.small_dataset) / self.args.batch_size)
        
    def Adjust_Learning_Rate(self):
        if self.mode == "Pytorch"   : 
            self.learning_rate = 0.0001
            if self.epoch > 60 and self.epoch < 90:
                self.learning_rate = 0.00001
            elif self.epoch > 90:
                self.learning_rate = 0.0000001
        if self.mode == "Python"    :
            self.learning_rate = 0.0001
            if self.epoch > 60 and self.epoch < 90:
                self.learning_rate = 0.00001
            elif self.epoch > 90:
                self.learning_rate = 0.0000001
        if self.mode == "Simulation": 
            pass #Add code by Thaising
        if self.mode == "FPGA"      : 
            # Code by GiTae 
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
        # self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.train_data_iter)
        if self.mode == "Pytorch"   :
            modtorch_model = DeepConvNetTorch(input_dims=(3, 416, 416),
                                            num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                            max_pools=[0, 1, 2, 3, 4],
                                            weight_scale='kaiming',
                                            batchnorm=True,
                                            dtype=torch.float32, device='cpu')
        if self.mode == "Python"    :
            python_model = DeepConvNet(input_dims=(3, 416, 416),
                                            num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                            max_pools=[0, 1, 2, 3, 4],
                                            weight_scale='kaiming',
                                            batchnorm=True,
                                            dtype=torch.float32, device='cpu')
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : 
            # Code by GiTae
            self.Image_1_start = time.time()   
            self.YOLOv2TinyFPGA = YOLOv2_Tiny_FPGA(self.Weight_Dec, self.Bias_Dec, 
                                    self.Beta_Dec, self.Gamma_Dec,
                                    self.Running_Mean_Dec, 
                                    self.Running_Var_Dec,
                                    self.im_data,
                                    self.Brain_Floating_Point, 
                                    self) 

            s = time.time()
            self.YOLOv2TinyFPGA.Write_Weight(self)       
            e = time.time()
            print("Write Weight Time : ",e-s)

            s = time.time()
            self.YOLOv2TinyFPGA.Write_Image()
            e = time.time()
            print("Write Image Time : ",e-s)

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

    def Forward(self):
        Input = self.im_data
        if self.mode == "Pytorch"   :
            X = Input
        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
            if self.batchnorm:
                for bn_params in self.bn_params:
                    bn_params['mode'] = 'test'

            scores = None
            # pass conv_param to the forward pass for the convolutional layer
            # Padding and stride chosen to preserve the input spatial size
            filter_size = 3
            conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

            scores = None
            # pass conv_param to the forward pass for the convolutional layer
            # Padding and stride chosen to preserve the input spatial size
            filter_size = 3
            conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

            scores = None

            slowpool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 1}
            cache = {}
            Out = {}
            self.phase = 'Forward'
            temp_Out = {}
            temp_cache = {}
            
            self.save_txt = False
            self.save_hex = False
            
            #0
            temp_Out[0], temp_cache['0'] = Torch_Conv_Pool.forward(X,
                                                        self.params['W0'],
                                                        conv_param,
                                                        pool_param,
                                                        layer_no=0,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[0], layer_no=0, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)

            Out[0], cache['0'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(X,
                                                        self.params['W0'],
                                                        self.params['gamma0'],
                                                        self.params['beta0'],
                                                        conv_param,
                                                        self.bn_params[0],
                                                        mean,
                                                        var,
                                                        pool_param,
                                                        layer_no=0,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )
            #1   
            temp_Out[1], temp_cache['1'] = Torch_FastConv.forward(Out[0],
                                                        self.params['W1'],
                                                        conv_param,
                                                        # pool_param,
                                                        layer_no=1,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[1], layer_no=1, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)

            Out[1], cache['1'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out[0],
                                                        self.params['W1'],
                                                        self.params['gamma1'],
                                                        self.params['beta1'],
                                                        conv_param,
                                                        self.bn_params[1],
                                                        mean,
                                                        var,
                                                        pool_param,
                                                        layer_no=1,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )
            #2
            temp_Out[2], temp_cache['2'] = Torch_FastConv.forward(Out[1],
                                                        self.params['W2'],
                                                        conv_param,
                                                        # pool_param,
                                                        layer_no=2,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[2], layer_no=2, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)

            Out[2], cache['2'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out[1],
                                                        self.params['W2'],
                                                        self.params['gamma2'],
                                                        self.params['beta2'],
                                                        conv_param,
                                                        self.bn_params[2],
                                                        mean,
                                                        var,
                                                        pool_param,
                                                        layer_no=2,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )
            #3
            temp_Out[3], temp_cache['3'] = Torch_FastConv.forward(Out[2],
                                                        self.params['W3'],
                                                        conv_param,
                                                        # pool_param,
                                                        layer_no=3,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[3], layer_no=3, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)

            Out[3], cache['3'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out[2],
                                                        self.params['W3'],
                                                        self.params['gamma3'],
                                                        self.params['beta3'],
                                                        conv_param,
                                                        self.bn_params[3],
                                                        mean,
                                                        var,
                                                        pool_param,
                                                        layer_no=3,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )

        #4
            temp_Out[4], temp_cache['4'] = Torch_FastConv.forward(Out[3],
                                                        self.params['W4'],
                                                        conv_param,
                                                        # pool_param,
                                                        layer_no=4,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[4], layer_no=4, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)

            Out[4], cache['4'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out[3],
                                                        self.params['W4'],
                                                        self.params['gamma4'],
                                                        self.params['beta4'],
                                                        conv_param,
                                                        self.bn_params[4],
                                                        mean,
                                                        var,
                                                        pool_param,
                                                        layer_no=4,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )
            
            #5    
            temp_Out[5], temp_cache['5'] = Torch_FastConv.forward(Out[4],
                                                        self.params['W5'],
                                                        conv_param,
                                                        layer_no=5,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[5], layer_no=5, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)


            Out[5], cache['5'] = Torch_Conv_BatchNorm_ReLU.forward(Out[4],
                                                        self.params['W5'],
                                                        self.params['gamma5'],
                                                        self.params['beta5'],
                                                        conv_param,
                                                        self.bn_params[5],
                                                        mean,
                                                        var,
                                                        layer_no=5,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )
            #6    
            temp_Out[6], temp_cache['6'] = Torch_FastConv.forward(Out[5],
                                                        self.params['W6'],
                                                        conv_param,
                                                        layer_no=6,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[6], layer_no=6, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)


            Out[6], cache['6'] = Torch_Conv_BatchNorm_ReLU.forward(Out[5],
                                                        self.params['W6'],
                                                        self.params['gamma6'],
                                                        self.params['beta6'],
                                                        conv_param,
                                                        self.bn_params[6],
                                                        mean,
                                                        var,
                                                        layer_no=6,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )

            #7    
            temp_Out[7], temp_cache['7'] = Torch_FastConv.forward(Out[6],
                                                        self.params['W7'],
                                                        conv_param,
                                                        layer_no=7,
                                                        save_txt=False,
                                                        save_hex=False,
                                                        phase=self.phase,
                                                        )
            
            mean, var = Cal_mean_var.forward(temp_Out[7], layer_no=7, save_txt=self.save_txt, save_hex=self.save_hex, phase=self.phase)


            Out[7], cache['7'] = Torch_Conv_BatchNorm_ReLU.forward(Out[6],
                                                        self.params['W7'],
                                                        self.params['gamma7'],
                                                        self.params['beta7'],
                                                        conv_param,
                                                        self.bn_params[7],
                                                        mean,
                                                        var,
                                                        layer_no=7,
                                                        save_txt=self.save_txt,
                                                        save_hex=self.save_hex,
                                                        phase=self.phase,
                                                        )
            
            #8
            conv_param['pad'] = 0
            temp_Out[8], temp_cache['8'] = Torch_FastConvWB.forward(Out[7],
                                                                self.params['W8'],
                                                                self.params['b8'],
                                                                conv_param,
                                                                layer_no=8,
                                                                save_txt=False,
                                                                save_hex=False,
                                                                phase=self.phase)

            mean, var = Cal_mean_var.forward(temp_Out[8], layer_no=8, save_txt=False, save_hex=False, phase=self.phase)

            Out[8], cache['8'] = Torch_FastConvWB.forward(Out[7],
                                                    self.params["W8"],
                                                    self.params["b8"],
                                                    conv_param,
                                                    layer_no=8,
                                                    save_txt=self.save_txt,
                                                    save_hex=self.save_hex,
                                                    phase=self.phase,
                                                    )


            out = Out[8]
            # print('\n\nFwd Out', out.dtype, out[out != 0], '\n\n')

            return out, cache, Out
            # pass #Add code by Wathna
        if self.mode == "Python"    : 
            if self.batchnorm:
                for bn_params in self.bn_params:
                    bn_params['mode'] = 'test'
            
            self.save_txt = False
            self.save_hex = False

            scores = None
            # pass conv_param to the forward pass for the convolutional layer
            # Padding and stride chosen to preserve the input spatial size
            filter_size = 3
            conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

            scores = None
            # pass conv_param to the forward pass for the convolutional layer
            # Padding and stride chosen to preserve the input spatial size
            filter_size = 3
            conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

            # pass pool_param to the forward pass for the max-pooling layer
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

            scores = None

            slowpool_param = {'pool_height':2, 'pool_width':2, 'stride': 1}
            cache = {}
            Out={}
            self.phase='Forward'
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   X = convert_to_16(self, X)
            Out[0], cache['0'] = Python_Conv_BatchNorm_ReLU_Pool.forward(X                      , 
                                                                        self.params['W0']       , 
                                                                        self.params['gamma0']   , 
                                                                        self.params['beta0']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[0]       ,
                                                                        pool_param              ,
                                                                        layer_no= 0             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            # Out[0] = convert_to_16(self, Out[0])
            Out[1], cache['1'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[0]                 , 
                                                                        self.params['W1']       , 
                                                                        self.params['gamma1']   , 
                                                                        self.params['beta1']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[1]       ,
                                                                        pool_param              ,
                                                                        layer_no= 1             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )

            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            # Out[1] = convert_to_16(self, Out[1])
            Out[2], cache['2'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[1]                 , 
                                                                        self.params['W2']       , 
                                                                        self.params['gamma2']   , 
                                                                        self.params['beta2']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[2]       ,
                                                                        pool_param              ,
                                                                        layer_no= 2             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[2] = convert_to_16(self, Out[2])
            Out[3], cache['3'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[2]                 , 
                                                                        self.params['W3']       , 
                                                                        self.params['gamma3']   , 
                                                                        self.params['beta3']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[3]       ,
                                                                        pool_param              ,
                                                                        layer_no= 3             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[3] = convert_to_16(self, Out[3])
            Out[4], cache['4'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[3]                 , 
                                                                        self.params['W4']       , 
                                                                        self.params['gamma4']   , 
                                                                        self.params['beta4']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[4]       ,
                                                                        pool_param              ,
                                                                        layer_no= 4             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[4] = convert_to_16(self, Out[4])
            Out[5], cache['5'] = Python_Conv_BatchNorm_ReLU.forward     (Out[4]                 , 
                                                                        self.params['W5']       , 
                                                                        self.params['gamma5']   , 
                                                                        self.params['beta5']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[5]       ,
                                                                        layer_no= 5             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            
            # save_file('Input' , Out[5], module='Pad', layer_no=6, save_hex=self.save_hex, save_txt=self.save_txt, phase=self.phase)
            # Out[60]            = F.pad                                  (Out[5] , (0, 1, 0, 1))
            # save_file('Output', Out[60], module='Pad', layer_no=6, save_hex=self.save_hex, save_txt=self.save_txt, phase=self.phase)
            
            # Out[61],cache['60']= Python_MaxPool.forward                 (Out[60]                , 
            #                                                             slowpool_param          ,
            #                                                             layer_no= 6            , 
            #                                                             save_txt= self.save_txt , 
            #                                                             save_hex= self.save_hex ,
            #                                                             phase   = self.phase    ,
            #                                                             )
            
            # Out[6], cache['6'] = Python_Conv_BatchNorm_ReLU.forward     (Out[61]                , 
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[5] = convert_to_16(self, Out[5])
            Out[6], cache['6'] = Python_Conv_BatchNorm_ReLU.forward     (Out[5]                , 
                                                                        self.params['W6']       , 
                                                                        self.params['gamma6']   , 
                                                                        self.params['beta6']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[6]       ,
                                                                        layer_no= 6             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[6] = convert_to_16(self, Out[6])
            Out[7], cache['7'] = Python_Conv_BatchNorm_ReLU.forward     (Out[6]                 , 
                                                                        self.params['W7']       , 
                                                                        self.params['gamma7']   , 
                                                                        self.params['beta7']    , 
                                                                        conv_param              , 
                                                                        self.bn_params[7]       ,
                                                                        layer_no= 7             , 
                                                                        save_txt= self.save_txt , 
                                                                        save_hex= self.save_hex ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[7] = convert_to_16(self, Out[7])
            conv_param['pad']  = 0
            Out[8], cache['8'] = Python_ConvB.forward                   (Out[7]                 , 
                                                                        self.params['W8']       , 
                                                                        self.params['b8']       , 
                                                                        conv_param              ,
                                                                        layer_no=8              , 
                                                                        save_txt=self.save_txt  , 
                                                                        save_hex=self.save_hex  ,
                                                                        phase   = self.phase    ,
                                                                        args = self  ,
                                                                        )
            
            # if self.convert_to_fp16 and self.convert_layer_IpOp and self.convert_forward: 
            #   Out[8] = convert_to_16(self, Out[8])
            out = Out[8]
            
            print('\n\nFwd Out', out.dtype, out[out!=0],'\n\n')
            
            return out, cache, Out
        if self.mode == "Simulation": # Add Code By Thaising
            Weight_Tensor = self.Weight_Dec
            Gamma_Tensor = self.Gamma_Dec
            Beta_Tensor = self.Beta_Dec
            bias = self.Bias_Dec
            running_mean = self.Running_Mean_Dec
            running_var = self.Running_Var_Dec
            filter_size = 3
            conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
            pool_param_stride2 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
            cache = {}
            temp_Out = {}
            temp_cache = {}

            # Layer0: Conv-BN-ReLU-Pool
            temp_Out[0], temp_cache['0'] = Torch_Conv_Pool.forward(Input, Weight_Tensor[0], conv_param, pool_param_stride2)
            
            mean, var = Cal_mean_var.forward(temp_Out[0])
            
            Out0, cache['0'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Input, Weight_Tensor[0], Gamma_Tensor[0],
                                                                    Beta_Tensor[0], conv_param, running_mean[0], 
                                                                    running_var[0], mean, var, self.Mode, pool_param_stride2)
            # Layer1: Conv-BN-ReLU-Pool
            temp_Out[1], temp_cache['1'] = Torch_FastConv.forward(Out0, Weight_Tensor[1], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[1])
            
            Out1, cache['1'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out0, Weight_Tensor[1], Gamma_Tensor[1], Beta_Tensor[1],
                                                                    conv_param, running_mean[1], running_var[1],
                                                                    mean, var, self.Mode, pool_param_stride2)
            # Layer2: Conv-BN-ReLU-Pool
            temp_Out[2], temp_cache['2'] = Torch_FastConv.forward(Out1, Weight_Tensor[2], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[2])
            
            Out2, cache['2'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out1, Weight_Tensor[2], Gamma_Tensor[2], Beta_Tensor[2],
                                                                    conv_param, running_mean[2], running_var[2],
                                                                    mean, var, self.Mode, pool_param_stride2)
            # Layer3: Conv-BN-ReLU-Pool
            temp_Out[3], temp_cache['3'] = Torch_FastConv.forward(Out2, Weight_Tensor[3], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[3])
            
            Out3, cache['3'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out2, Weight_Tensor[3], Gamma_Tensor[3], Beta_Tensor[3],
                                                                    conv_param, running_mean[3], running_var[3],
                                                                    mean, var, self.Mode, pool_param_stride2)
            # Layer4: Conv-BN-ReLU-Pool
            temp_Out[4], temp_cache['4'] = Torch_FastConv.forward(Out3, Weight_Tensor[4], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[4])
            
            Out4, cache['4'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out3, Weight_Tensor[4], Gamma_Tensor[4], Beta_Tensor[4],
                                                                    conv_param, running_mean[4], running_var[4],
                                                                    mean, var, self.Mode, pool_param_stride2)
            # Layer5: Conv-BN-ReLU
            temp_Out[5], temp_cache['5'] = Torch_FastConv.forward(Out4, Weight_Tensor[5], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[5])
            
            Out5, cache['5'] = Torch_Conv_BatchNorm_ReLU.forward(Out4, Weight_Tensor[5], Gamma_Tensor[5], Beta_Tensor[5],
                                                                conv_param, running_mean[5], running_var[5],
                                                                mean, var, self.Mode)

            # Layer6: Conv-BN-ReLU
            temp_Out[6], temp_cache['6'] = Torch_FastConv.forward(Out5, Weight_Tensor[6], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[6])
            
            Out6, cache['6'] = Torch_Conv_BatchNorm_ReLU.forward(Out5, Weight_Tensor[6], Gamma_Tensor[6],
                                                                Beta_Tensor[6], conv_param, running_mean[6], running_var[6],
                                                                mean, var, self.Mode)
            # Layer7: Conv-BN-ReLU
            temp_Out[7], temp_cache['7'] = Torch_FastConv.forward(Out6, Weight_Tensor[7], conv_param)
            
            mean, var = Cal_mean_var.forward(temp_Out[7])
            
            Out7, cache['7'] = Torch_Conv_BatchNorm_ReLU.forward(Out6, Weight_Tensor[7], Gamma_Tensor[7], Beta_Tensor[7],
                                                                conv_param, running_mean[7], running_var[7],
                                                                mean, var, self.Mode)
            # Layer8: ConvWB
            conv_param['pad'] = 0
            Out8, cache['8'] = Torch_FastConvWB.forward(Out7, Weight_Tensor[8], bias, conv_param)
            Output_Image = Out8
            self.Output_Image, self.cache = Output_Image, cache 
            # return Output_Image, cache
        
        if self.mode == "FPGA"      : 
            # Code by GiTae 
            self.YOLOv2TinyFPGA = YOLOv2_Tiny_FPGA(self.Weight_Dec, self.Bias_Dec, 
                                self.Beta_Dec, self.Gamma_Dec,
                                self.Running_Mean_Dec, 
                                self.Running_Var_Dec,
                                self.im_data,
                                self) 
            print("Start NPU")
            s = time.time()
            self.YOLOv2TinyFPGA.Forward(self)
            e = time.time()
            print("Forward Process Time : ",e-s)
            self.change_color_red()
            # return Bias_Grad
    
    def Calculate_Loss(self):

        if self.mode == "Pytorch"   :
            print('Calculating the loss and its gradients for pytorch model.')
            out = torch.tensor(out, requires_grad=True)

            scores = out
            bsize, _, h, w = out.shape
            out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

            xy_pred = torch.sigmoid(out[:, :, 0:2])
            conf_pred = torch.sigmoid(out[:, :, 4:5])
            hw_pred = torch.exp(out[:, :, 2:4])
            class_score = out[:, :, 5:]
            class_pred = F.softmax(class_score, dim=-1)
            delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)


            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [v for v in target_data]

            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)
            _loss = box_loss + iou_loss + class_loss
            
            print(f"\nLoss = {_loss}\n")
            out = scores
            out.retain_grad()
            _loss.backward(retain_graph=True)
            dout = out.grad.detach()
            # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
            # dout = pickle.load(dout)
            # print('\n\n',dout.dtype, dout[dout!=0])
            print(f'\n\nLoss Gradients\n\t{dout.dtype}\n\t{dout[dout!=0][:10]}')
            
            # # Save output for circuit team and pickle for future.
            # if self.save_pickle:
            #   Path("Temp_Files/Python").mkdir(parents=True, exist_ok=True)
            #   with open('Temp_Files/Python/Backward_loss_gradients.pickle','wb') as handle:
            #     pickle.dump(dout,handle, protocol=pickle.HIGHEST_PROTOCOL)
            # if self.save_output:
            #   Path("Outputs/Python/Backward/").mkdir(parents=True, exist_ok=True)
            #   save_file(f'Outputs/Python/Backward/Backward_loss_gradients.txt', dout)
            #   # save_file(f'Outputs/Python/Backward/Loss.txt', loss)
                
            # if self.convert_to_fp16 and self.convert_loss_grad:
            #   dout = convert_to_16(self, dout)
            #   loss = convert_to_16(self, loss)
            return _loss, dout
        
        if self.mode == "Python"    :
            print('Calculating the loss and its gradients for python model.')
            out = torch.tensor(out, requires_grad=True)

            scores = out
            bsize, _, h, w = out.shape
            out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

            xy_pred = torch.sigmoid(out[:, :, 0:2])
            conf_pred = torch.sigmoid(out[:, :, 4:5])
            hw_pred = torch.exp(out[:, :, 2:4])
            class_score = out[:, :, 5:]
            class_pred = F.softmax(class_score, dim=-1)
            delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)


            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [v for v in target_data]

            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)
            _loss = box_loss + iou_loss + class_loss
            
            print(f"\nLoss = {_loss}\n")
            out = scores
            out.retain_grad()
            _loss.backward(retain_graph=True)
            dout = out.grad.detach()
            # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
            # dout = pickle.load(dout)
            # print('\n\n',dout.dtype, dout[dout!=0])
            print(f'\n\nLoss Gradients\n\t{dout.dtype}\n\t{dout[dout!=0][:10]}')
            
            return _loss, dout
        
        if self.mode == "Simulation": # Add By Thaising
                
            OutImage_Data=self.Output_Image
            gt_boxes=self.gt_boxes
            gt_classes=self.gt_classes
            num_boxes=self.num_obj
            
            if self.Mode == "Training":
                self.Loss, self.Loss_Gradient = loss(out=OutImage_Data, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_boxes)
                # return Loss, Loss_Gradient
            if self.Mode == "Inference":
                self.output_data = reshape_output(gt_boxes=None, gt_classes=None, num_boxes=None)
                # return output_data
        if self.mode == "FPGA"      : 
            self.Loss, self.Loss_Gradient = self.YOLOv2TinyFPGA.Post_Processing(self, gt_boxes=self.gt_boxes, gt_classes=self.gt_classes, num_boxes=self.num_obj)

       
    def Before_Backward(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : 
            self.YOLOv2TinyFPGA.Pre_Processing_Backward(self, self.Loss_Gradient)
     
    def Backward(self):

        if self.mode == "Pytorch"   :
            grads = {}
            dOut = {}
            self.save_txt = False
            self.save_hex = False
            self.phase = 'Backwards'
            
            dOut[8], grads['W8'], grads['b8'] = Torch_FastConvWB.backward(dout,
                                                                    cache['8'],
                                                                    layer_no=8,
                                                                    save_txt=True,
                                                                    save_hex=True,
                                                                    phase=self.phase)
            dw, db = grads['W8'], grads['b8'] 
            last_dout = dOut[8]
            
            dOut[7], grads['W7'], grads['gamma7'], grads['beta7'] = Torch_Conv_BatchNorm_ReLU.backward(
                dOut[8],
                cache['7'],
                layer_no=7,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

        

            dOut[6], grads['W6'], grads['gamma6'], grads['beta6'] = Torch_Conv_BatchNorm_ReLU.backward(
                dOut[7],
                cache['6'],
                layer_no=6,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            dOut[5], grads['W5'], grads['gamma5'], grads['beta5'] = Torch_Conv_BatchNorm_ReLU.backward(
                dOut[6],
                cache['5'],
                layer_no=5,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            dOut[4], grads['W4'], grads['gamma4'], grads['beta4'] = Torch_Conv_BatchNorm_ReLU_Pool.backward(
                dOut[5],
                cache['4'],
                layer_no=4,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            dOut[3], grads['W3'], grads['gamma3'], grads['beta3'] = Torch_Conv_BatchNorm_ReLU_Pool.backward(
                dOut[4],
                cache['3'],
                layer_no=3,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            dOut[2], grads['W2'], grads['gamma2'], grads['beta2'] = Torch_Conv_BatchNorm_ReLU_Pool.backward(
                dOut[3],
                cache['2'],
                layer_no=2,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            dOut[1], grads['W1'], grads['gamma1'], grads['beta1'] = Torch_Conv_BatchNorm_ReLU_Pool.backward(
                dOut[2],
                cache['1'],
                layer_no=1,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            dOut[0], grads['W0'], grads['gamma0'], grads['beta0'] = Torch_Conv_BatchNorm_ReLU_Pool.backward(
                dOut[1],
                cache['0'],
                layer_no=0,
                save_txt=self.save_txt,
                save_hex=self.save_hex,
                phase=self.phase,
            )

            return dOut, grads
        if self.mode == "Python"    :
            grads = {}
            dOut={}
            self.save_hex = False
            self.save_txt = False
            self.phase ='Backwards'

            dOut[8], grads['W8'], grads['b8']                     = Python_ConvB.backward(                dout,  
                                                                                                        cache['8'], 
                                                                                                        layer_no=8              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,)
                
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[8]     = convert_to_16(self, dOut[8])
            #   grads['W8'] = convert_to_16(self, grads['W8'])
            #   grads['b8'] = convert_to_16(self, grads['b8'])
            
            
            # last_dout = 2 * last_dout
            # dw        = 2 * dw
            # db        = 2 * db

            # dw, db = grads['W8'], grads['b8']
            # print(f'\n\tdw8\n\t\t{dw.shape}\n\t\t{dw[dw!=0]}\n\t\t{dw.dtype}')
            # print(f'\n\tdb8\n\t\t{db.shape}\n\t\t{db[db!=0]}\n\t\t{db.dtype}')
            # print(f'\n\tlast_dout\n\t\t{dOut[8].shape}\n\t\t{dOut[8][dOut[8]!=0]}\n\t\t{dOut[8].dtype}')

            # print(f'\n\nBackward Grads Outputs')   
            # print(f"\n\t grads['W8']\n\t\t{grads['W8'].shape}\n\t\t{grads['W8'][grads['W8']!=0]}\n")


            dOut[7], grads['W7'], grads['gamma7'], grads['beta7']  = Python_Conv_BatchNorm_ReLU.backward( 
                                                                                                        dOut[8]                 , 
                                                                                                        cache['7']              , 
                                                                                                        layer_no=7              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[7]     = convert_to_16(self, dOut[7])
            #   grads['W7'] = convert_to_16(self, grads['W7'])
            #   grads['gamma7'] = convert_to_16(self, grads['gamma7'])
            #   grads['beta7'] = convert_to_16(self, grads['beta7'])


            dOut[6], grads['W6'], grads['gamma6'], grads['beta6']  = Python_Conv_BatchNorm_ReLU.backward( 
                                                                                                        dOut[7]                 , 
                                                                                                        cache['6']              , 
                                                                                                        layer_no=6              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[6]     = convert_to_16(self, dOut[6])
            #   grads['W6'] = convert_to_16(self, grads['W6'])
            #   grads['gamma6'] = convert_to_16(self, grads['gamma6'])
            #   grads['beta6'] = convert_to_16(self, grads['beta6'])


            dOut[5], grads['W5'], grads['gamma5'], grads['beta5']  = Python_Conv_BatchNorm_ReLU.backward( 
                                                                                                        dOut[6]                 , 
                                                                                                        cache['5']              , 
                                                                                                        layer_no=5              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[5]     = convert_to_16(self, dOut[5])
            #   grads['W5'] = convert_to_16(self, grads['W5'])
            #   grads['gamma5'] = convert_to_16(self, grads['gamma5'])
            #   grads['beta5'] = convert_to_16(self, grads['beta5'])


            dOut[4], grads['W4'], grads['gamma4'], grads['beta4']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                        dOut[5]                 , 
                                                                                                        cache['4']              , 
                                                                                                        layer_no=4              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[4]     = convert_to_16(self, dOut[4])
            #   grads['W4'] = convert_to_16(self, grads['W4'])
            #   grads['gamma4'] = convert_to_16(self, grads['gamma4'])
            #   grads['beta4'] = convert_to_16(self, grads['beta4'])


            dOut[3], grads['W3'], grads['gamma3'], grads['beta3']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                        dOut[4]                 , 
                                                                                                        cache['3']              , 
                                                                                                        layer_no=3              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[3]     = convert_to_16(self, dOut[3])
            #   grads['W3'] = convert_to_16(self, grads['W3'])
            #   grads['gamma3'] = convert_to_16(self, grads['gamma3'])
            #   grads['beta3'] = convert_to_16(self, grads['beta3'])


            dOut[2], grads['W2'], grads['gamma2'], grads['beta2']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                        dOut[3]                 , 
                                                                                                        cache['2']              , 
                                                                                                        layer_no=2              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[2]     = convert_to_16(self, dOut[2])
            #   grads['W2'] = convert_to_16(self, grads['W2'])
            #   grads['gamma2'] = convert_to_16(self, grads['gamma2'])
            #   grads['beta2'] = convert_to_16(self, grads['beta2'])


            dOut[1], grads['W1'], grads['gamma1'], grads['beta1']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                        dOut[2]                 , 
                                                                                                        cache['1']              , 
                                                                                                        layer_no=1              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[1]     = convert_to_16(self, dOut[1])
            #   grads['W1'] = convert_to_16(self, grads['W1'])
            #   grads['gamma1'] = convert_to_16(self, grads['gamma1'])
            #   grads['beta1'] = convert_to_16(self, grads['beta1'])


            dOut[0], grads['W0'], grads['gamma0'], grads['beta0']  = Python_Conv_BatchNorm_ReLU_Pool.backward( 
                                                                                                        dOut[1]                 , 
                                                                                                        cache['0']              , 
                                                                                                        layer_no=0              , 
                                                                                                        save_txt=self.save_txt  , 
                                                                                                        save_hex=self.save_hex  ,
                                                                                                        phase   = self.phase    ,
                                                                                                        args = self ,
                                                                                                        )
            # if self.convert_to_fp16 and self.convert_backward and self.convert_layer_IpOp:
            #   dOut[0]     = convert_to_16(self, dOut[0])
            #   grads['W0'] = convert_to_16(self, grads['W0'])
            #   grads['gamma0'] = convert_to_16(self, grads['gamma0'])
            #   grads['beta0'] = convert_to_16(self, grads['beta0'])


            
            # # Sa =     convert_to_16(self, 
            # # Sa

                
            # # Save pickle files for future use
            # if self.save_pickle:
            #   Path("Temp_Files/Python/").mkdir(parents=True, exist_ok=True)
            #   with open('Temp_Files/Python/Backward_dOut.pickle','wb') as handle:
            #     pickle.dump(dOut,handle, protocol=pickle.HIGHEST_PROTOCOL)
            #   with open('Temp_Files/Python/Backward_grads.pickle','wb') as handle:
            #     pickle.dump(grads,handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # if self.save_output:
            #   Path("Outputs/Python/Backward/").mkdir(parents=True, exist_ok=True)
            #   for _key in dOut.keys():
            #     save_file(f'Outputs/Python/Backward/dOut_Layer_{_key}.txt', dOut[_key])
            #   for _key in grads.keys():
            #     save_file(f'Outputs/Python/Backward/grads_Layer_{_key}.txt', grads[_key])
                
                
            return  dOut, grads
        if self.mode == "Simulation": # Add By Thaising
            Loss_Gradient, cache = self.Loss_Gradient, self.cache
            Input_Grad_Layer8, Weight_Gradient_Layer8, Bias_Grad  = Torch_FastConvWB.backward(Loss_Gradient, cache['8'])
            Input_Grad_Layer7, Weight_Gradient_Layer7, Gamma_Gradient_Layer7, Beta_Gradient_Layer7  = Torch_Conv_BatchNorm_ReLU.backward (Input_Grad_Layer8, cache['7'])
            Input_Grad_Layer6, Weight_Gradient_Layer6, Gamma_Gradient_Layer6, Beta_Gradient_Layer6  = Torch_Conv_BatchNorm_ReLU.backward (Input_Grad_Layer7, cache['6'])
            Input_Grad_Layer5, Weight_Gradient_Layer5, Gamma_Gradient_Layer5, Beta_Gradient_Layer5  = Torch_Conv_BatchNorm_ReLU.backward (Input_Grad_Layer6, cache['5'])
            Input_Grad_Layer4, Weight_Gradient_Layer4, Gamma_Gradient_Layer4, Beta_Gradient_Layer4  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer5, cache['4'])
            Input_Grad_Layer3, Weight_Gradient_Layer3, Gamma_Gradient_Layer3, Beta_Gradient_Layer3  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer4, cache['3'])
            Input_Grad_Layer2, Weight_Gradient_Layer2, Gamma_Gradient_Layer2, Beta_Gradient_Layer2  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer3, cache['2'])
            Input_Grad_Layer1, Weight_Gradient_Layer1, Gamma_Gradient_Layer1, Beta_Gradient_Layer1  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer2, cache['1'])
            Input_Grad_Layer0, Weight_Gradient_Layer0, Gamma_Gradient_Layer0, Beta_Gradient_Layer0  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer1, cache['0'])
            
            # Gradient Value for Weight Update
            self.Weight_Gradient = [Weight_Gradient_Layer0, Weight_Gradient_Layer1, Weight_Gradient_Layer2, Weight_Gradient_Layer3, 
                            Weight_Gradient_Layer4, Weight_Gradient_Layer5, Weight_Gradient_Layer6, Weight_Gradient_Layer7, 
                            Weight_Gradient_Layer8]
            
            self.Beta_Gradient = [Beta_Gradient_Layer0, Beta_Gradient_Layer1, Beta_Gradient_Layer2, Beta_Gradient_Layer3, 
                            Beta_Gradient_Layer4, Beta_Gradient_Layer5,Beta_Gradient_Layer6, Beta_Gradient_Layer7]
            
            self.Gamma_Gradient = [Gamma_Gradient_Layer0, Gamma_Gradient_Layer1, Gamma_Gradient_Layer2, Gamma_Gradient_Layer3, 
                            Gamma_Gradient_Layer4, Gamma_Gradient_Layer5, Gamma_Gradient_Layer6, Gamma_Gradient_Layer7]
            
            self.Bias_Grad = Bias_Grad
            
            # return Weight_Gradient, Bias_Grad, Gamma_Gradient, Beta_Gradient
        
        if self.mode == "FPGA"      : 
            s = time.time()
            self.YOLOv2TinyFPGA.Backward()
            e = time.time()
            print("Backward Process Time : ",e-s)

            self.change_color_red()

#         if self.mode == "FPGA"      : 
#             self.YOLOv2TinyFPGA = YOLOv2_Tiny_FPGA(self.Weight_Dec, self.Bias_Dec, 
#                                     self.Beta_Dec, self.Gamma_Dec,
#                                     self.Running_Mean_Dec, 
#                                     self.Running_Var_Dec,
#                                     self.im_data,
#                                     self) 
#             s = time.time()
#             self.Weight_Gradient, self.Bias_Grad, self.Beta_Gradient, self.Gamma_Gradient = self.YOLOv2TinyFPGA.Backward(self, self.Loss_Gradient)
#             e = time.time()
#             print("Backward Process Time : ",e-s)

#             self.change_color_red()
#             # return Weight_Gradient, Beta_Gradient, Gamma_Gradient

    def Weight_Update(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": # Add By Thaising
            [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec], self.custom_model = \
            self.Shoaib.update_weights_FPGA(
                Inputs  = [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec], 
                gInputs = [self.Weight_Gradient,  self.Bias_Grad,  self.Gamma_Gradient, self.Beta_Gradient ])
        if self.mode == "FPGA"      : 
            s = time.time()
            # Modified By Thaising
            Shoaib = Shoaib_Code(
                Weight_Dec=self.Weight_Dec, 
                Bias_Dec=self.Bias_Dec, 
                Beta_Dec=self.Beta_Dec, 
                Gamma_Dec=self.Gamma_Dec,
                Running_Mean_Dec=self.Running_Mean_Dec, 
                Running_Var_Dec=self.Running_Var_Dec,
                args=self.args,
                pth_weights_path="/home/msis/training/yolov2/src/Pre_Processing_Scratch/data/pretrained/yolov2_best_map.pth",
                model=Yolov2,
                optim=optim)
            
            # [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec], self.custom_model = \
            [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec] = \
            Shoaib.update_weights_FPGA(
                Inputs  = [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec], 
                gInputs = [self.Weight_Gradient,  self.Bias_Grad,  self.Gamma_Gradient, self.Beta_Gradient ])
            e = time.time()
            print("Weight Update Time : ", e-s)
            self.Image_1_end = time.time()
            print("1 Image Train Time : ",self.Image_1_end-self.Image_1_start)
            # self.output_text = f"Batch: {step+1}/{10}--Loss: {Loss}"
            # print(f"Batch: {step+1}/{10}--Loss: {Loss}")
            # self.Show_Text(self.output_text)

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
    
#         # Save Pickle: 
#         if self.epoch % self.args.save_interval == 0:
#             self._data = self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec, self.epoch
#             self.output_file = os.path.join(self.args.output_dir, f'Params_{self.epoch}.pickle')
#             with open(self.output_file, 'wb') as handle:
#                 pickle.dump(self._data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    def Check_mAP(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": 
            self.map = self.Shoaib.cal_mAP(Inputs_with_running = \
                [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, self.Running_Mean_Dec, self.Running_Var_Dec])
        if self.mode == "FPGA"      : 
            Shoaib = Shoaib_Code(
                Weight_Dec=self.Weight_Dec, 
                Bias_Dec=self.Bias_Dec, 
                Beta_Dec=self.Beta_Dec, 
                Gamma_Dec=self.Gamma_Dec,
                Running_Mean_Dec=self.Running_Mean_Dec, 
                Running_Var_Dec=self.Running_Var_Dec,
                args=self.args,
                pth_weights_path="/home/msis/training/yolov2/src/Pre_Processing_Scratch/data/pretrained/yolov2_best_map.pth",
                model=Yolov2,
                optim=optim)
            self.map = Shoaib.cal_mAP(Inputs_with_running = [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, 
                                                             self.Running_Mean_Dec, self.Running_Var_Dec])
    
    def Post_Epoch(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": # Add By Thaising
            self.whole_process_end = time.time()
            self.whole_process_time = self.whole_process_end - self.whole_process_start
            print(f'\n\t---------------------Best mAP was at Epoch {self.best_map_epoch}, with mAP={self.best_map_score}% and loss={self.best_map_loss}\n')
        if self.mode == "FPGA"      :     
            self.whole_process_end = time.time()
            self.whole_process_time = self.whole_process_end - self.whole_process_start
            self.output_text = f"Epoch: {self.epoch+1}/{self.args.max_epochs}--Loss: {self.Loss}"
            print(f"Epoch: {self.epoch}/{self.args.max_epochs}--Loss: {self.Loss}")
            self.Show_Text(self.output_text)
            print(f"Epoch: {self.epoch}/{self.args.max_epochs}--Loss: {self.Loss}")
        
        
if __name__ == "__main__":
    app = App()
    app.mainloop()

