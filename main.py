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

import warnings
warnings.filterwarnings("ignore")

from tkinter.constants import DISABLED, NORMAL 

import os
import sys
sys.path.append("../")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"Dataset"))
sys.path.append(os.path.join(os.getcwd(),"src"))
sys.path.append(os.path.join(os.getcwd(),"src/Main_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Pre_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Post_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Weight_Update_Algorithm"))
sys.path.append(os.path.join(os.getcwd(),"Codes"))
sys.path.append("/home/msis/Desktop/pcie_python/GUI")
from  XdmaAccess import XdmaAccess
from Pre_Processing_Scratch.Pre_Processing import *
from Pre_Processing_Scratch.Pre_Processing_Function import *
from Post_Processing_Scratch.Post_Processing_2Iterations import Post_Processing
from Pre_Processing_Scratch.ImageLoader import ImageLoader
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


        # configure window
        self.title("Yolov2 Accelerator.py")
        self.geometry(f"{165+145+10+600-120+10+170+10+170+10}x{855-20}")
        
        
        #customtkinter.set_appearance_mode("Dark")
        #top_frame
        self.top_frame = customtkinter.CTkFrame(self, width=1140, height=96)
        self.top_frame.place(x=20, y=20)
        
        try: original_image = Image.open("1.png")
        except: original_image = Image.open("Codes/1.png")
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

        #microcode = Microcode("mic_2iteration_forward_hex_add_0x.txt") 
        microcode = Microcode("MICROCODE.txt")

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
        print(f"Start Training")
        self.Show_Text(f"Start Training")
        
        print(f'Mode : {self.mode}')
        
        self.parse_args()
        
        self.Pre_Process()
        
        self.Create_Output_Dir()
        
        self.Load_Weights()
        
        self.Load_Dataset()

        for epoch in range(self.args.start_epoch, self.args.max_epochs):
            self.curr_epoch = epoch
            
            self.Adjust_Learning_Rate()
            
            self.whole_process_start = time.time()
            self.train_data_iter = iter(self.small_dataloader)
            
            for step in range(self.iters_per_epoch):
                self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.train_data_iter)
                
                self.Before_Forward()
                self.Forward()
                self.Loss()
                self.Before_Backward()
                self.Backward()
                self.Weight_Update() 

                if epoch % 18 == 0:
                    self.Clear_Text()

            self.Save_Pickle()
            
            self.Check_mAP()
            
            self.Post_Epoch()

    def Run_Infer(self):
        self.Train.configure(state="disabled")
        self.Infer.configure(state="disabled")
        self.Infer.configure(fg_color='green')
        self.Stop.configure(state="normal")
        self.Stop.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        print(f"Start Inference")
        self.Show_Text(f"Start Inference")
        pass      
        
    def Stop_Process(self):
        self.Train.configure(state="normal")
        self.Train.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Infer.configure(state="normal")
        self.Infer.configure(fg_color=['#3B8ED0', '#1F6AA5'])
        self.Stop.configure(state="disabled")
        self.Stop.configure(fg_color='green')
        print("Stop the process")
        self.Show_Text(f"Stop the process")
        
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
                            default=80, type=int)
        parser.add_argument('--total_inference_set', dest='total_inference_set',
                            default=10, type=int)
        parser.add_argument('--batch_size', dest='batch_size',
                            default=8, type=int)
        parser.add_argument('--nw', dest='num_workers',
                            help='number of workers to load training data',
                            default=2, type=int)
        parser.add_argument('--save_interval', dest='save_interval',
                            default=10, type=int)

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
        
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        
         # Create Directory
        if not os.path.exists(out_path):
            os.makedirs(out_path)

    def Load_Weights(self):
        
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        
        s = time.time()
        self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec = self.PreProcessing.WeightLoader()
        e = time.time()
        print("WeightLoader : ",e-s)
        
        pass

    def Load_Dataset(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
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
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        # learning_rate = 0.001
        self.learning_rate = 0.001
        # Various of Learning will Change with the Epochs    
        if self.curr_epoch >= 10 and self.curr_epoch < 20:
            self.learning_rate = 0.0001
        elif self.curr_epoch >= 20 and self.curr_epoch < 30:
            self.learning_rate = 0.000001
        elif self.curr_epoch >= 30:
            self.learning_rate = 0.0000001
            
    def Before_Forward(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass

        self.Image_1_start = time.time()   
        self.YOLOv2TinyFPGA = YOLOv2_Tiny_FPGA(self.Weight_Dec, self.Bias_Dec, 
                                self.Beta_Dec, self.Gamma_Dec,
                                self.Running_Mean_Dec, 
                                self.Running_Var_Dec,
                                self.im_data,
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
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass

        print("Start NPU")
        
        s = time.time()
        self.Loss = self.YOLOv2TinyFPGA.Forward(gt_boxes=self.gt_boxes, gt_classes=self.gt_classes, num_boxes=self.num_obj)
        e = time.time()
        print("Forward Process Time : ",e-s)
        self.change_color_red()
    
    def Loss(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        pass
       
    def Before_Backward(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        pass
     
    def Backward(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        
        s = time.time()
        self.YOLOv2TinyFPGA.Backward()
        e = time.time()
        print("Backward Process Time : ",e-s)

        self.change_color_red()

    def Weight_Update(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        pass

        s = time.time()
        self.Inputs  = self.Weight_Dec,       self.Bias_Dec,   self.Gamma_Dec,      self.Beta_Dec,
        self.gInputs = self.Weight_Gradient,  self.Bias_Grad,  self.Gamma_Gradient, self.Beta_Gradient
        self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec = self.YOLOv2TinyFPGA.Weight_Update(self.Inputs, self.gInputs, self.learning_rate)
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
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        pass
    
        # Save Pickle: 
        if self.curr_epoch % self.args.save_interval == 0:
            self._data = self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec, self.curr_epoch
            self.output_file = os.path.join(self.output_dir, f'Params_{self.curr_epoch}.pickle')
            with open(self.output_file, 'wb') as handle:
                pickle.dump(self._data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    def Check_mAP(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass
        pass
    
    def Post_Epoch(self):
        if self.mode == "Pytorch"   : pass
        if self.mode == "Python"    : pass
        if self.mode == "Simulation": pass
        if self.mode == "FPGA"      : pass    
        self.whole_process_end = time.time()
        self.whole_process_time = self.whole_process_end - self.whole_process_start
        # print("1 epoch time : ",whole_process_time)
        # print("epoch : ", epoch+1, ", Loss : ", Loss)
        self.output_text = f"Epoch: {self.curr_epoch+1}/{self.args.max_epochs}--Loss: {self.Loss}"
        print(f"Epoch: {self.curr_epoch}/{self.args.max_epochs}--Loss: {self.Loss}")
        self.Show_Text(self.output_text)
        print(f"Epoch: {self.curr_epoch}/{self.args.max_epochs}--Loss: {self.Loss}")
        
        
if __name__ == "__main__":
    app = App()
    app.mainloop()
