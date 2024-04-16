import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from pypcie import Device
from ast import literal_eval
import subprocess
from  XdmaAccess import XdmaAccess
import sys
sys.path.append("../")
from Pre_Processing_Scratch.Pre_Processing import Pre_Processing, Read_OutFmap_Bfloat2Dec, Cal_mean_var, Mean_Var_Dec2Bfloat, \
                                                Loss_Gradient_Dec2Bfloat, Read_WeightGradient_Bfloat2Dec, Break_FlipHex_256To32
from Pre_Processing_Scratch.Pre_Processing_Function import *
from Post_Processing_Scratch.Post_Processing_2Iterations import Post_Processing
import time
from tabulate import tabulate
import os.path 


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

MAX_LINE_LENGTH = 1000

# Floating Point Parameters that we use
FloatingPoint_Format = "FP32"

# For the BFP16, we don't need to select format, we just use Single Precision, Truncated and Rounding

Selected_FP_Format = {
    "FP32": (8, 23),
    "Bfloat16": (5, 10),
    "Custom": (7, 8)
}

Exponent_Bits, Mantissa_Bits = Selected_FP_Format.get(FloatingPoint_Format, (0, 0))

# Pre-define Conditions:
Mode = "Training"  # Declare whether is Training or Inference

# Pre-Processing Pre-Defined Conditions
Brain_Floating_Point    = True  # Declare Whether Bfloat16 Conversion or not
# Signals for Debugging Purposes
Image_Converted         = True  # Converted Images
Weight_Converted        = True  # Converted Weight
Bias_Converted          = True  # Converted Bias
BN_Param_Converted      = True  # Converted BN Parameters
Microcode               = True  # Microcode Writing

# Running Hardware Model:
YOLOv2_Hardware_Forward     = True
YOLOv2_Hardware_Backward    = True

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("CustomTkinter complex_example.py")
        self.geometry(f"{1190}x{855}")
        
        
        #customtkinter.set_appearance_mode("Dark")
        #top_frame
        # 이미지를 로드합니다.
        self.top_frame = customtkinter.CTkFrame(self, width=1140, height=96)
        self.top_frame.place(x=20, y=20)
        
        image_path = "1.png"
        original_image = Image.open(image_path)
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
        
        #left_frame
        self.left_frame = customtkinter.CTkFrame(self, width=145, height=680)  # 너비 값 조정
        self.left_frame.place(x=20, y=140)
        
        self.left_label = customtkinter.CTkLabel(self.left_frame, text="Control Panel", font=("Helvetica", 15))
        self.left_label.place(x=25, y=10)               
        

        background_color = (255, 255, 255)  # White background color
        #left_frame_button
        button_width = 120
        button_height = 30
        self.Load_PCIe  = customtkinter.CTkButton(self.left_frame, text="Load PCIe", command=self.Load_PCIe_click, width=button_width, height=button_height)
        self.Load_PCIe.place(x=10, y=50)

        self.Load_Data = customtkinter.CTkButton(self.left_frame, text="Load Data", command=self.Load_Data_click, width=button_width, height=button_height)
        self.Load_Data.place(x=10, y=100)
        
        self.Load_Microcode = customtkinter.CTkButton(self.left_frame, text="Load Microcode", command=self.Load_Microcode_click, width=button_width, height=button_height)
        self.Load_Microcode.place(x=10, y=150)

        self.Run= customtkinter.CTkButton(self.left_frame, text="Run ", command=self.Run_click, width=button_width, height=button_height)
        self.Run .place(x=10, y=200)
        
        self.Stop = customtkinter.CTkButton(self.left_frame, text="Stop", command=self.Stop_click, width=button_width, height=button_height)
        self.Stop.place(x=10, y=250)

        self.Weight_Update = customtkinter.CTkButton(self.left_frame, text="Weight Update", command=self.Weight_Update_click, width=button_width, height=button_height)
        self.Weight_Update.place(x=10, y=330)
        
        self.Reset = customtkinter.CTkButton(self.left_frame, text="Reset", command=self.Reset_click, width=button_width, height=button_height)
        self.Reset.place(x=10, y=380)

       

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=600, height=680)  # Adjust the height value as needed
        self.textbox.place(x=185, y=140)

        

        # right_frame_1
        self.right_frame_1 = customtkinter.CTkFrame(self, width=170, height=680)
        self.right_frame_1.place(x=800, y=140)
        
        
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
        self.right_frame_2.place(x=990, y=140)
        
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

        
    def Load_PCIe_click(self):
        print("Load PCIe Driver!")
        exit_code = subprocess.call(["./load_driver.sh"], cwd='./t_cbu/test2')
    
    def Load_Data_click(self):
        print("Button 2 clicked")

        self.d = Device("0000:08:00.0")
        # Access BAR 0
        self.bar = self.d.bar[2]
        i = 0x0
        j = 0x0
        k = 0x0
        l = 0x0 
        #with open("32bit_flip_line/YOLOv2_Weight_32_flip_8.txt") as file:
        with open("Weight_CH0.txt") as file:
          for item in file:
            #print(item) 
            an_integer = int(item, 16)
           # hex_value = hex(an_integer)
            self.bar.write(0x0000 +i, an_integer)
            #print(an_integer)
            i = i + 0x4

        print("Weight0 Done!")
 
        #with open("32bit_flip_line/YOLOv2_Image_32_flip_8.txt") as file:
        with open("Image_CH0.txt") as file:
          for item in file:
            #print(item) 
            an_integer = int(item, 16)
           # hex_value = hex(an_integer)
            self.bar.write(0x2400000 +j, an_integer)
            #print(an_integer)
            j = j + 0x4

        print("Image0 Done!")

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
        print("Button 3 clicked")
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]
        #self.textbox.insert("0.0", "CTkTextbox\n\n" )

        microcode = Microcode("mic_2iteration_Forward_hex_add_0x.txt")

        for i in range (0, len(microcode)):
            self.bar.write(0x4, microcode[i]) # wr mic
            self.bar.write(0x8, i) # wr addr
            self.bar.write(0x0, 0x00000012) # wr en
            self.bar.write(0x0, 0x00000010) # wr en low
        print("mic write done")    
        
        
    def Run_click(self):
        # start signal
        print("Button 5 clicked")
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


        start = time.time()
        #################################################
        #                Layer 0 Start                  #
        #################################################       
        # layer0 capture interrupt
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            if os.path.isfile(output_file_name_2):
                with open(output_file_name_1, "r") as file1, \
                    open(output_file_name_2, "r") as file2:
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                        while ch1 and ch2:
                            if ch1 != ch2:
                                print("layer0 interrupt1: 1")
                                irq_val = 1
                                self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")
                            ch1 = file1.read(1)
                            ch2 = file2.read(1)


                        if irq_val != 1:
                            print("layer0 interrupt1: 0")

                        with open(output_file_name_1, "rb") as file1, \
                            open(output_file_name_2, "wb") as file2:

                            buffer = file1.read(MAX_LINE_LENGTH)
                            while buffer:
                                file2.write(buffer)
                                buffer = file1.read(MAX_LINE_LENGTH)

                        print("Done")
                        file1.close()
                        file2.close()
            else:  
                with open(input_file_name, "r") as input_file, \
                    open(output_file_name_1, "w") as output_file:

                    for line in input_file:
                        if "xdma" in line:
                            output_file.write(line)
                            if " 1 " in line:
                                irq_val=1
                                self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")

                                print("layer0 interrupt0: 1")
                            else:
                                irq_val=0
                                print("layer0 interrupt0: 0") 

                    input_file.close()
                    output_file.close()            

                    if irq_val == 1:
                        with open(output_file_name_1, "rb") as file1, \
                            open(output_file_name_2, "wb") as file2:

                            buffer = file1.read(MAX_LINE_LENGTH)
                            while buffer:
                                file2.write(buffer)
                                buffer = file1.read(MAX_LINE_LENGTH)    

                        file1.close()
                        file2.close()        

            print("extract xdma line Done!\n")
            '''
            if i == 1:
                with open(output_file_name_1, "rb") as file1, \
                     open(output_file_name_2, "wb") as file2:

                    buffer = file1.read(MAX_LINE_LENGTH)
                    while buffer:
                        file2.write(buffer)
                        buffer = file1.read(MAX_LINE_LENGTH)    
                                         
                #i=i+1
                print("copy Done")
                input_file.close()       
                output_file.close()
                file1.close()
                file2.close()
            else :
                with open(output_file_name_1, "r") as file1, \
                    open(output_file_name_2, "r") as file2:
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                        while ch1 and ch2:
                            if ch1 != ch2:
                                print("interrupt1: 1")
                                irq_val = 1
                            ch1 = file1.read(1)
                            ch2 = file2.read(1)

                        if irq_val != 1:
                            print("interrupt1: 0")

                        with open(output_file_name_1, "rb") as file1, \
                            open(output_file_name_2, "wb") as file2:

                            buffer = file1.read(MAX_LINE_LENGTH)
                            while buffer:
                                file2.write(buffer)
                                buffer = file1.read(MAX_LINE_LENGTH)
                
                        print("Done")
                        file1.close()
                        file2.close()
                '''        
        # Pre-Processing Class Initialization
        
        PreProcessing = Pre_Processing(Mode                 =   Mode,
                                    Brain_Floating_Point =   Brain_Floating_Point,
                                    Exponent_Bits        =   Exponent_Bits,
                                    Mantissa_Bits        =   Mantissa_Bits)   
        if Bias_Converted:
            Bias_List = PreProcessing.Bias_Converted_Func()
        
        if BN_Param_Converted:
            Beta_List = PreProcessing.Beta_Param_Converted_Func()
            Gamma_List = PreProcessing.Gamma_Param_Converted_Func()
            RunningMean_List = PreProcessing.Running_Mean_Param_Converted_Func()
            RunningVar_List = PreProcessing.Running_Var_Param_Converted_Func()
            
        if Weight_Converted:
            print("\t --> " + " The Weights are processing, Please Wait for a Moment!")
            Weight_List = PreProcessing.Weight_Converted_Func()
        
        # Layer 0
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer0_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X3E00000, End_Address=0X3ED0000)
        Layer0_1st_Iter_Image1_CH0_256 = process_input_lines(Layer0_1st_Iter_Image1_CH0)
        print("ch0 image 1 : ", len(Layer0_1st_Iter_Image1_CH0)) 

        Layer0_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X3ED0000, End_Address=0X3FA0000)
        Layer0_1st_Iter_Image2_CH0_256 = process_input_lines(Layer0_1st_Iter_Image2_CH0)
        print("ch0 image 2 : ", len(Layer0_1st_Iter_Image2_CH0))
       
        Layer0_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X3FA0000, End_Address=0X4070000)
        Layer0_1st_Iter_Image3_CH0_256 = process_input_lines(Layer0_1st_Iter_Image3_CH0)
        print("ch0 image 3 : ", len(Layer0_1st_Iter_Image3_CH0))

        Layer0_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X4070000, End_Address=0X4140000)
        Layer0_1st_Iter_Image4_CH0_256 = process_input_lines(Layer0_1st_Iter_Image4_CH0)
        print("ch0 image 4 : ", len(Layer0_1st_Iter_Image4_CH0))

        Layer0_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X4140000, End_Address=0X4210000)
        Layer0_1st_Iter_Image5_CH0_256 = process_input_lines(Layer0_1st_Iter_Image5_CH0)
        print("ch0 image 5 : ", len(Layer0_1st_Iter_Image5_CH0))

        Layer0_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X4210000, End_Address=0X42E0000)
        Layer0_1st_Iter_Image6_CH0_256 = process_input_lines(Layer0_1st_Iter_Image6_CH0)
        print("ch0 image 6 : ", len(Layer0_1st_Iter_Image6_CH0))

        Layer0_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X42E0000, End_Address=0X43B0000)
        Layer0_1st_Iter_Image7_CH0_256 = process_input_lines(Layer0_1st_Iter_Image7_CH0)
        print("ch0 image 7 : ", len(Layer0_1st_Iter_Image7_CH0))

        Layer0_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X43B0000, End_Address=0X4480000)
        Layer0_1st_Iter_Image8_CH0_256 = process_input_lines(Layer0_1st_Iter_Image8_CH0)
        print("ch0 image 8 : ", len(Layer0_1st_Iter_Image8_CH0))


        Layer0_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X13E00000, End_Address=0X13ED0000)
        Layer0_1st_Iter_Image1_CH1_256 = process_input_lines(Layer0_1st_Iter_Image1_CH1)
        print("ch1 image 1 : ", len(Layer0_1st_Iter_Image1_CH1))

        Layer0_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X13ED0000, End_Address=0X13FA0000)
        Layer0_1st_Iter_Image2_CH1_256 = process_input_lines(Layer0_1st_Iter_Image2_CH1)
        print("ch1 image 2 : ", len(Layer0_1st_Iter_Image2_CH1))

        Layer0_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X13FA0000, End_Address=0X14070000)
        Layer0_1st_Iter_Image3_CH1_256 = process_input_lines(Layer0_1st_Iter_Image3_CH1)
        print("ch1 image 3 : ", len(Layer0_1st_Iter_Image3_CH1))

        Layer0_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X14070000, End_Address=0X14140000)
        Layer0_1st_Iter_Image4_CH1_256 = process_input_lines(Layer0_1st_Iter_Image4_CH1)
        print("ch1 image 4 : ", len(Layer0_1st_Iter_Image4_CH1))

        Layer0_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X14140000, End_Address=0X14210000)
        Layer0_1st_Iter_Image5_CH1_256 = process_input_lines(Layer0_1st_Iter_Image5_CH1)
        print("ch1 image 5 : ", len(Layer0_1st_Iter_Image5_CH1))

        Layer0_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X14210000, End_Address=0X142E0000)
        Layer0_1st_Iter_Image6_CH1_256 = process_input_lines(Layer0_1st_Iter_Image6_CH1)
        print("ch1 image 6 : ", len(Layer0_1st_Iter_Image6_CH1))

        Layer0_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X142E0000, End_Address=0X143B0000)
        Layer0_1st_Iter_Image7_CH1_256 = process_input_lines(Layer0_1st_Iter_Image7_CH1)
        print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Layer0_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X143B0000, End_Address=0X14480000)
        Layer0_1st_Iter_Image8_CH1_256 = process_input_lines(Layer0_1st_Iter_Image8_CH1)
        print("ch1 image 8 : ", len(Layer0_1st_Iter_Image8_CH1))

        '''
        test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        print("Convert Format")
        Output_Image1_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image1_CH0_256, Layer0_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image2_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image2_CH0_256, Layer0_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image3_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image3_CH0_256, Layer0_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image4_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image4_CH0_256, Layer0_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image5_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image5_CH0_256, Layer0_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image6_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image6_CH0_256, Layer0_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image7_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image7_CH0_256, Layer0_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image8_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image8_CH0_256, Layer0_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)

        OutImages_1st_Layer0 = Output_Image1_Layer0_1st_Iter + Output_Image2_Layer0_1st_Iter + Output_Image3_Layer0_1st_Iter + Output_Image4_Layer0_1st_Iter + \
                            Output_Image5_Layer0_1st_Iter + Output_Image6_Layer0_1st_Iter + Output_Image7_Layer0_1st_Iter + Output_Image8_Layer0_1st_Iter    

        OutImage_1st_Layer0 = torch.tensor([float(value) for value in OutImages_1st_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer0, Var_1st_Layer0 = Cal_mean_var.Forward(OutImage_1st_Layer0)
        Mean_1st_Layer0, Var_1st_Layer0 = Mean_Var_Dec2Bfloat(Mean_1st_Layer0, Var_1st_Layer0, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, Weight_List[0], Mean_1st_Layer0, Var_1st_Layer0, Beta_List[0], Iteration="2")
        '''
        data_read_mean_var = "result/Mean_1st_Layer0.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Mean_1st_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close() 

        data_read_mean_var = "result/Var_1st_Layer0.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Var_1st_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close() 
        
        data_read_mean_var = "result/layer0_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Weight_2nd_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close()               
        '''
        # Write DDR
        print("Write DDR")

        weight_layer0_2nd_ch0 = Break_FlipHex_256To32(Weight_2nd_Layer0[0])
        weight_layer0_2nd_ch1 = Break_FlipHex_256To32(Weight_2nd_Layer0[1])

        data_read_mean_var = "result/weight_layer0_2nd_ch0.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in weight_layer0_2nd_ch0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")      
        output_file.close()  

        data_read_mean_var = "result/weight_layer0_2nd_ch1.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in weight_layer0_2nd_ch1:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write

        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer0[0]), Wr_Address=0x0)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer0[1]), Wr_Address=0x10000000)
        
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)
        print("layer0 end")

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer0_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 
        
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]
        
        data_read = open("result/layer0_result_ch0_image1.txt", mode="w+")
        i=0
        for i in range(0,int((0X4550000-0X4480000)/4) ): 
            Read_Data = self.bar.read(0X4480000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer0_result_ch1_image1.txt", mode="w+")
        i=0
        for i in range(0,int((0X4550000-0X4480000)/4) ): 
            Read_Data = self.bar.read(0X14480000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     

        data_read = open("result/layer0_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X784C000-0X71CC000)/4) ): 
            Read_Data = self.bar.read(0X71CC000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''
            
        #################################################
        #                Layer 1 Start                  #
        #################################################
        # check Layer1 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer1 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer1 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close()
            
            print("extract xdma line Done!\n")              

        # Layer 1
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer1_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X4B00000, End_Address=0X4CA0000)
        Layer1_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer1_1st_Iter_Image1_CH0))    
        
        Layer1_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X4CA0000, End_Address=0X4E40000)
        Layer1_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer1_1st_Iter_Image2_CH0))
       
        Layer1_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X4E40000, End_Address=0X4FE0000)
        Layer1_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer1_1st_Iter_Image3_CH0))

        Layer1_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X4FE0000, End_Address=0X5180000)
        Layer1_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer1_1st_Iter_Image4_CH0))

        Layer1_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X5180000, End_Address=0X5320000)
        Layer1_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer1_1st_Iter_Image5_CH0))

        Layer1_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X5320000, End_Address=0X54C0000)
        Layer1_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer1_1st_Iter_Image6_CH0))

        Layer1_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X54C0000, End_Address=0X5660000)
        Layer1_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer1_1st_Iter_Image7_CH0))

        Layer1_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X5660000, End_Address=0X5800000)
        Layer1_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer1_1st_Iter_Image8_CH0))


        Layer1_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X14B00000, End_Address=0X14CA0000)
        Layer1_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer1_1st_Iter_Image1_CH1))

        Layer1_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X14CA0000, End_Address=0X14E40000)
        Layer1_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer1_1st_Iter_Image2_CH1))

        Layer1_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X14E40000, End_Address=0X14FE0000)
        Layer1_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer1_1st_Iter_Image3_CH1))

        Layer1_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X14FE0000, End_Address=0X15180000)
        Layer1_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer1_1st_Iter_Image4_CH1))

        Layer1_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X15180000, End_Address=0X15320000)
        Layer1_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer1_1st_Iter_Image5_CH1))

        Layer1_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X15320000, End_Address=0X154C0000)
        Layer1_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer1_1st_Iter_Image6_CH1))

        Layer1_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X154C0000, End_Address=0X15660000)
        Layer1_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer1_1st_Iter_Image7_CH1))

        Layer1_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X15660000, End_Address=0X15800000)
        Layer1_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer1_1st_Iter_Image8_CH1))


        test_out = '1st_iter_result/Layer1_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        print("Convert Format")
        Output_Image1_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image1_CH0_256, Layer1_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image2_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image2_CH0_256, Layer1_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image3_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image3_CH0_256, Layer1_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image4_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image4_CH0_256, Layer1_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image5_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image5_CH0_256, Layer1_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image6_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image6_CH0_256, Layer1_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image7_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image7_CH0_256, Layer1_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image8_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image8_CH0_256, Layer1_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)

        OutImages_1st_Layer1 = Output_Image1_Layer1_1st_Iter + Output_Image2_Layer1_1st_Iter + Output_Image3_Layer1_1st_Iter + Output_Image4_Layer1_1st_Iter + \
                            Output_Image5_Layer1_1st_Iter + Output_Image6_Layer1_1st_Iter + Output_Image7_Layer1_1st_Iter + Output_Image8_Layer1_1st_Iter    

        OutImage_1st_Layer1 = torch.tensor([float(value) for value in OutImages_1st_Layer1], dtype=torch.float32).reshape(8, 32, 208, 208)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer1, Var_1st_Layer1 = Cal_mean_var.Forward(OutImage_1st_Layer1)
        Mean_1st_Layer1, Var_1st_Layer1 = Mean_Var_Dec2Bfloat(Mean_1st_Layer1, Var_1st_Layer1, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, Weight_List[1], Mean_1st_Layer1, Var_1st_Layer1, Beta_List[1], Iteration="2")
        
        data_read_mean_var = "result/layer1_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer1:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")      
        output_file.close()          
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer1[0]), Wr_Address=0xA00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer1[1]), Wr_Address=0x10000A00)
        
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, 1)
        print(irq_val)
        
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)
        print("layer1 end")

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer1_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer1_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X5B40000-0X5800000)/4) ): 
            Read_Data = self.bar.read(0X5800000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer1_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X5B40000-0X5800000)/4) ): 
            Read_Data = self.bar.read(0X15800000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer1_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7B8C000-0X784C000)/4) ): 
            Read_Data = self.bar.read(0X784C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''   

        #################################################
        #                Layer 2 Start                  #
        #################################################
        # check Layer2 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer2 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L2_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer2 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close()       

            print("extract xdma line Done!\n")

        
        # Layer 2
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer2_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X5B40000, End_Address=0X5C10000)
        Layer2_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer2_1st_Iter_Image1_CH0))     

        Layer2_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X5C10000, End_Address=0X5CE0000)
        Layer2_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer2_1st_Iter_Image2_CH0))
       
        Layer2_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X5CE0000, End_Address=0X5DB0000)
        Layer2_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer2_1st_Iter_Image3_CH0))

        Layer2_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X5DB0000, End_Address=0X5E80000)
        Layer2_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer2_1st_Iter_Image4_CH0))

        Layer2_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X5E80000, End_Address=0X5F50000)
        Layer2_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer2_1st_Iter_Image5_CH0))

        Layer2_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X5F50000, End_Address=0X6020000)
        Layer2_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer2_1st_Iter_Image6_CH0))

        Layer2_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X6020000, End_Address=0X60F0000)
        Layer2_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer2_1st_Iter_Image7_CH0))

        Layer2_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X60F0000, End_Address=0X61C0000)
        Layer2_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer2_1st_Iter_Image8_CH0))


        Layer2_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X15B40000, End_Address=0X15C10000)
        Layer2_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer2_1st_Iter_Image1_CH1))

        Layer2_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X15C10000, End_Address=0X15CE0000)
        Layer2_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer2_1st_Iter_Image2_CH1))

        Layer2_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X15CE0000, End_Address=0X15DB0000)
        Layer2_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer2_1st_Iter_Image3_CH1))

        Layer2_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X15DB0000, End_Address=0X15E80000)
        Layer2_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer2_1st_Iter_Image4_CH1))

        Layer2_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X15E80000, End_Address=0X15F50000)
        Layer2_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer2_1st_Iter_Image5_CH1))

        Layer2_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X15F50000, End_Address=0X16020000)
        Layer2_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer2_1st_Iter_Image6_CH1))

        Layer2_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X16020000, End_Address=0X160F0000)
        Layer2_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer2_1st_Iter_Image7_CH1))

        Layer2_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X160F0000, End_Address=0X161C0000)
        Layer2_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer2_1st_Iter_Image8_CH1))


        test_out = '1st_iter_result/Layer2_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        print("Convert Format")
        Output_Image1_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image1_CH0_256, Layer2_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image2_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image2_CH0_256, Layer2_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image3_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image3_CH0_256, Layer2_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image4_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image4_CH0_256, Layer2_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image5_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image5_CH0_256, Layer2_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image6_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image6_CH0_256, Layer2_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image7_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image7_CH0_256, Layer2_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image8_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image8_CH0_256, Layer2_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)

        OutImages_1st_Layer2 = Output_Image1_Layer2_1st_Iter + Output_Image2_Layer2_1st_Iter + Output_Image3_Layer2_1st_Iter + Output_Image4_Layer2_1st_Iter + \
                            Output_Image5_Layer2_1st_Iter + Output_Image6_Layer2_1st_Iter + Output_Image7_Layer2_1st_Iter + Output_Image8_Layer2_1st_Iter    

        OutImage_1st_Layer2 = torch.tensor([float(value) for value in OutImages_1st_Layer2], dtype=torch.float32).reshape(8, 64, 104, 104)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer2, Var_1st_Layer2 = Cal_mean_var.Forward(OutImage_1st_Layer2)
        Mean_1st_Layer2, Var_1st_Layer2 = Mean_Var_Dec2Bfloat(Mean_1st_Layer2, Var_1st_Layer2, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, Weight_List[2], Mean_1st_Layer2, Var_1st_Layer2, Beta_List[2], Iteration="2")
        
        data_read_mean_var = "result/layer2_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer2:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")     
        output_file.close()           
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer2[0]), Wr_Address=0x1E00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer2[1]), Wr_Address=0x10001E00)
        
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print("layer2 end")

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer_2_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer2_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6360000-0X61C0000)/4) ): 
            Read_Data = self.bar.read(0X61C0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer2_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6360000-0X61C0000)/4) ): 
            Read_Data = self.bar.read(0X161C0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")

        data_read = open("result/layer2_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7D2C000-0X7B8C000)/4) ): 
            Read_Data = self.bar.read(0X7B8C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''


        #################################################
        #                Layer 3 Start                  #
        #################################################
        # check Layer3 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer3 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L3_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer3 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close()        

            print("extract xdma line Done!\n")

        # Layer 3
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer3_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X6360000, End_Address=0X63C8000)
        Layer3_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer3_1st_Iter_Image1_CH0))     

        Layer3_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X63C8000, End_Address=0X6430000)
        Layer3_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer3_1st_Iter_Image2_CH0))
       
        Layer3_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X6430000, End_Address=0X6498000)
        Layer3_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer3_1st_Iter_Image3_CH0))

        Layer3_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X6498000, End_Address=0X6500000)
        Layer3_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer3_1st_Iter_Image4_CH0))

        Layer3_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X6500000, End_Address=0X6568000)
        Layer3_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer3_1st_Iter_Image5_CH0))

        Layer3_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X6568000, End_Address=0X65D0000)
        Layer3_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer3_1st_Iter_Image6_CH0))

        Layer3_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X65D0000, End_Address=0X6638000)
        Layer3_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer3_1st_Iter_Image7_CH0))

        Layer3_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X6638000, End_Address=0X66A0000)
        Layer3_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer3_1st_Iter_Image8_CH0))


        Layer3_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X16360000, End_Address=0X163C8000)
        Layer3_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer3_1st_Iter_Image1_CH1))

        Layer3_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X163C8000, End_Address=0X16430000)
        Layer3_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer3_1st_Iter_Image2_CH1))

        Layer3_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X16430000, End_Address=0X16498000)
        Layer3_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer3_1st_Iter_Image3_CH1))

        Layer3_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X16498000, End_Address=0X16500000)
        Layer3_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer3_1st_Iter_Image4_CH1))

        Layer3_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X16500000, End_Address=0X16568000)
        Layer3_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer3_1st_Iter_Image5_CH1))

        Layer3_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X16568000, End_Address=0X165D0000)
        Layer3_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer3_1st_Iter_Image6_CH1))

        Layer3_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X165D0000, End_Address=0X16638000)
        Layer3_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer3_1st_Iter_Image7_CH1))

        Layer3_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X16638000, End_Address=0X166A0000)
        Layer3_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer3_1st_Iter_Image8_CH1))


        test_out = '1st_iter_result/Layer3_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()


        print("Convert Format")
        Output_Image1_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image1_CH0_256, Layer3_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image2_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image2_CH0_256, Layer3_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image3_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image3_CH0_256, Layer3_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image4_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image4_CH0_256, Layer3_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image5_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image5_CH0_256, Layer3_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image6_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image6_CH0_256, Layer3_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image7_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image7_CH0_256, Layer3_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image8_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image8_CH0_256, Layer3_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)

        OutImages_1st_Layer3 = Output_Image1_Layer3_1st_Iter + Output_Image2_Layer3_1st_Iter + Output_Image3_Layer3_1st_Iter + Output_Image4_Layer3_1st_Iter + \
                            Output_Image5_Layer3_1st_Iter + Output_Image6_Layer3_1st_Iter + Output_Image7_Layer3_1st_Iter + Output_Image8_Layer3_1st_Iter    

        OutImage_1st_Layer3 = torch.tensor([float(value) for value in OutImages_1st_Layer3], dtype=torch.float32).reshape(8, 128, 52, 52)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer3, Var_1st_Layer3 = Cal_mean_var.Forward(OutImage_1st_Layer3)
        Mean_1st_Layer3, Var_1st_Layer3 = Mean_Var_Dec2Bfloat(Mean_1st_Layer3, Var_1st_Layer3, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, Weight_List[3], Mean_1st_Layer3, Var_1st_Layer3, Beta_List[3], Iteration="2")
        
        data_read_mean_var = "result/layer3_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer3:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer3[0]), Wr_Address=0x6E00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer3[1]), Wr_Address=0x10006E00)
        
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer3_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer3_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6770000-0X66A0000)/4) ): 
            Read_Data = self.bar.read(0X66A0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer3_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6770000-0X66A0000)/4) ): 
            Read_Data = self.bar.read(0X166A0000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer3_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7DFC000-0X7D2C000)/4) ): 
            Read_Data = self.bar.read(0X7D2C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''

        #################################################
        #                Layer 4 Start                  #
        #################################################
        # check Layer4 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer4 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L4_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer4 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close()      

            print("extract xdma line Done!\n")

        # Layer 4
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer4_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X6770000, End_Address=0X67A4000)
        Layer4_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer4_1st_Iter_Image1_CH0))     

        Layer4_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X67A4000, End_Address=0X67D8000)
        Layer4_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer4_1st_Iter_Image2_CH0))
       
        Layer4_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X67D8000, End_Address=0X680C000)
        Layer4_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer4_1st_Iter_Image3_CH0))

        Layer4_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X680C000, End_Address=0X6840000)
        Layer4_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer4_1st_Iter_Image4_CH0))

        Layer4_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X6840000, End_Address=0X6874000)
        Layer4_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer4_1st_Iter_Image5_CH0))

        Layer4_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X6874000, End_Address=0X68A8000)
        Layer4_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer4_1st_Iter_Image6_CH0))

        Layer4_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X68A8000, End_Address=0X68DC000)
        Layer4_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer4_1st_Iter_Image7_CH0))

        Layer4_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X68DC000, End_Address=0X6910000)
        Layer4_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer4_1st_Iter_Image8_CH0))


        Layer4_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X16770000, End_Address=0X167A4000)
        Layer4_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer4_1st_Iter_Image1_CH1))

        Layer4_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X167A4000, End_Address=0X167D8000)
        Layer4_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer4_1st_Iter_Image2_CH1))

        Layer4_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X167D8000, End_Address=0X1680C000)
        Layer4_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer4_1st_Iter_Image3_CH1))

        Layer4_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X1680C000, End_Address=0X16840000)
        Layer4_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer4_1st_Iter_Image4_CH1))

        Layer4_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X16840000, End_Address=0X16874000)
        Layer4_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer4_1st_Iter_Image5_CH1))

        Layer4_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X16874000, End_Address=0X168A8000)
        Layer4_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer4_1st_Iter_Image6_CH1))

        Layer4_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X168A8000, End_Address=0X168DC000)
        Layer4_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer4_1st_Iter_Image7_CH1))

        Layer4_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X168DC000, End_Address=0X16910000)
        Layer4_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer4_1st_Iter_Image8_CH1))

        test_out = '1st_iter_result/Layer4_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()


        print("Convert Format")
        Output_Image1_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image1_CH0_256, Layer4_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image2_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image2_CH0_256, Layer4_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image3_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image3_CH0_256, Layer4_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image4_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image4_CH0_256, Layer4_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image5_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image5_CH0_256, Layer4_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image6_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image6_CH0_256, Layer4_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image7_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image7_CH0_256, Layer4_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image8_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image8_CH0_256, Layer4_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)

        OutImages_1st_Layer4 = Output_Image1_Layer4_1st_Iter + Output_Image2_Layer4_1st_Iter + Output_Image3_Layer4_1st_Iter + Output_Image4_Layer4_1st_Iter + \
                            Output_Image5_Layer4_1st_Iter + Output_Image6_Layer4_1st_Iter + Output_Image7_Layer4_1st_Iter + Output_Image8_Layer4_1st_Iter    

        OutImage_1st_Layer4 = torch.tensor([float(value) for value in OutImages_1st_Layer4], dtype=torch.float32).reshape(8, 256, 26, 26)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer4, Var_1st_Layer4 = Cal_mean_var.Forward(OutImage_1st_Layer4)
        Mean_1st_Layer4, Var_1st_Layer4 = Mean_Var_Dec2Bfloat(Mean_1st_Layer4, Var_1st_Layer4, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, Weight_List[4], Mean_1st_Layer4, Var_1st_Layer4, Beta_List[4], Iteration="2")
        
        data_read_mean_var = "result/layer4_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer4:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer4[0]), Wr_Address=0x1AE00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer4[1]), Wr_Address=0x1001AE00)
        
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer4_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer4_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6978000-0X6910000)/4) ): 
            Read_Data = self.bar.read(0X6910000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer4_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6978000-0X6910000)/4) ): 
            Read_Data = self.bar.read(0X16910000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer4_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7E64000-0X7DFC000)/4) ): 
            Read_Data = self.bar.read(0X7DFC000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''

        #################################################
        #                Layer 5 Start                  #
        #################################################
        # check Layer5 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer5 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L5_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer5 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close()

            print("extract xdma line Done!\n")

        # Layer 5
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer5_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X6978000, End_Address=0X6992000)
        Layer5_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer5_1st_Iter_Image1_CH0))     

        Layer5_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X6992000, End_Address=0X69AC000)
        Layer5_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer5_1st_Iter_Image2_CH0))
       
        Layer5_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X69AC000, End_Address=0X69C6000)
        Layer5_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer5_1st_Iter_Image3_CH0))

        Layer5_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X69C6000, End_Address=0X69E0000)
        Layer5_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer5_1st_Iter_Image4_CH0))

        Layer5_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X69E0000, End_Address=0X69FA000)
        Layer5_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer5_1st_Iter_Image5_CH0))

        Layer5_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X69FA000, End_Address=0X6A14000)
        Layer5_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer5_1st_Iter_Image6_CH0))

        Layer5_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X6A14000, End_Address=0X6A2E000)
        Layer5_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer5_1st_Iter_Image7_CH0))

        Layer5_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X6A2E000, End_Address=0X6A48000)
        Layer5_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer5_1st_Iter_Image8_CH0))


        Layer5_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X16978000, End_Address=0X16992000)
        Layer5_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer5_1st_Iter_Image1_CH1))

        Layer5_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X16992000, End_Address=0X169AC000)
        Layer5_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer5_1st_Iter_Image2_CH1))

        Layer5_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X169AC000, End_Address=0X169C6000)
        Layer5_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer5_1st_Iter_Image3_CH1))

        Layer5_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X169C6000, End_Address=0X169E0000)
        Layer5_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer5_1st_Iter_Image4_CH1))

        Layer5_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X169E0000, End_Address=0X169FA000)
        Layer5_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer5_1st_Iter_Image5_CH1))

        Layer5_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X169FA000, End_Address=0X16A14000)
        Layer5_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer5_1st_Iter_Image6_CH1))

        Layer5_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X16A14000, End_Address=0X16A2E000)
        Layer5_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer5_1st_Iter_Image7_CH1))

        Layer5_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X16A2E000, End_Address=0X16A48000)
        Layer5_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer5_1st_Iter_Image8_CH1))


        test_out = '1st_iter_result/Layer5_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()


        print("Convert Format")
        Output_Image1_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image1_CH0_256, Layer5_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image2_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image2_CH0_256, Layer5_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image3_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image3_CH0_256, Layer5_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image4_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image4_CH0_256, Layer5_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image5_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image5_CH0_256, Layer5_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image6_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image6_CH0_256, Layer5_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image7_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image7_CH0_256, Layer5_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image8_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image8_CH0_256, Layer5_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)

        OutImages_1st_Layer5 = Output_Image1_Layer5_1st_Iter + Output_Image2_Layer5_1st_Iter + Output_Image3_Layer5_1st_Iter + Output_Image4_Layer5_1st_Iter + \
                            Output_Image5_Layer5_1st_Iter + Output_Image6_Layer5_1st_Iter + Output_Image7_Layer5_1st_Iter + Output_Image8_Layer5_1st_Iter    

        OutImage_1st_Layer5 = torch.tensor([float(value) for value in OutImages_1st_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer5, Var_1st_Layer5 = Cal_mean_var.Forward(OutImage_1st_Layer5)
        Mean_1st_Layer5, Var_1st_Layer5 = Mean_Var_Dec2Bfloat(Mean_1st_Layer5, Var_1st_Layer5, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, Weight_List[5], Mean_1st_Layer5, Var_1st_Layer5, Beta_List[5], Iteration="2")
        
        data_read_mean_var = "result/layer5_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer5:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer5[0]), Wr_Address=0x6AE00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer5[1]), Wr_Address=0x1006AE00)
        
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer5_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer5_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
            Read_Data = self.bar.read(0X6A48000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer5_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
            Read_Data = self.bar.read(0X16A48000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer5_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7F34000-0X7E64000)/4) ): 
            Read_Data = self.bar.read(0X7E64000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''

        #################################################
        #                Layer 6 Start                  #
        #################################################
        # check Layer6 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer6 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L6_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer6 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close() 

            print("extract xdma line Done!\n")

        # Layer 6
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer6_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X6B18000, End_Address=0X6B4C000)
        Layer6_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer6_1st_Iter_Image1_CH0))     

        Layer6_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X6B4C000, End_Address=0X6B80000)
        Layer6_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer6_1st_Iter_Image2_CH0))
       
        Layer6_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X6B80000, End_Address=0X6BB4000)
        Layer6_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer6_1st_Iter_Image3_CH0))

        Layer6_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X6BB4000, End_Address=0X6BE8000)
        Layer6_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer6_1st_Iter_Image4_CH0))

        Layer6_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X6BE8000, End_Address=0X6C1C000)
        Layer6_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer6_1st_Iter_Image5_CH0))

        Layer6_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X6C1C000, End_Address=0X6C50000)
        Layer6_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer6_1st_Iter_Image6_CH0))

        Layer6_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X6C50000, End_Address=0X6C84000)
        Layer6_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer6_1st_Iter_Image7_CH0))

        Layer6_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X6C84000, End_Address=0X6CB8000)
        Layer6_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer6_1st_Iter_Image8_CH0))


        Layer6_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X16B18000, End_Address=0X16B4C000)
        Layer6_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer6_1st_Iter_Image1_CH1))

        Layer6_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X16B4C000, End_Address=0X16B80000)
        Layer6_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer6_1st_Iter_Image2_CH1))

        Layer6_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X16B80000, End_Address=0X16BB4000)
        Layer6_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer6_1st_Iter_Image3_CH1))

        Layer6_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X16BB4000, End_Address=0X16BE8000)
        Layer6_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer6_1st_Iter_Image4_CH1))

        Layer6_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X16BE8000, End_Address=0X16C1C000)
        Layer6_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer6_1st_Iter_Image5_CH1))

        Layer6_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X16C1C000, End_Address=0X16C50000)
        Layer6_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer6_1st_Iter_Image6_CH1))

        Layer6_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X16C50000, End_Address=0X16C84000)
        Layer6_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer6_1st_Iter_Image7_CH1))

        Layer6_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X16C84000, End_Address=0X16CB8000)
        Layer6_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer6_1st_Iter_Image8_CH1))


        test_out = '1st_iter_result/Layer6_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()


        print("Convert Format")
        Output_Image1_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image1_CH0_256, Layer6_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image2_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image2_CH0_256, Layer6_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image3_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image3_CH0_256, Layer6_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image4_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image4_CH0_256, Layer6_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image5_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image5_CH0_256, Layer6_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image6_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image6_CH0_256, Layer6_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image7_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image7_CH0_256, Layer6_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image8_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image8_CH0_256, Layer6_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)

        OutImages_1st_Layer6 = Output_Image1_Layer6_1st_Iter + Output_Image2_Layer6_1st_Iter + Output_Image3_Layer6_1st_Iter + Output_Image4_Layer6_1st_Iter + \
                            Output_Image5_Layer6_1st_Iter + Output_Image6_Layer6_1st_Iter + Output_Image7_Layer6_1st_Iter + Output_Image8_Layer6_1st_Iter    

        OutImage_1st_Layer6 = torch.tensor([float(value) for value in OutImages_1st_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer6, Var_1st_Layer6 = Cal_mean_var.Forward(OutImage_1st_Layer6)
        Mean_1st_Layer6, Var_1st_Layer6 = Mean_Var_Dec2Bfloat(Mean_1st_Layer6, Var_1st_Layer6, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, Weight_List[6], Mean_1st_Layer6, Var_1st_Layer6, Beta_List[6], Iteration="2")
        
        data_read_mean_var = "result/layer6_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer6:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer6[0]), Wr_Address=0x1AAE00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer6[1]), Wr_Address=0x101AAE00)
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer6_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer6_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
            Read_Data = self.bar.read(0X6CB8000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer6_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
            Read_Data = self.bar.read(0X16CB8000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer6_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X80D4000-0X7F34000)/4) ): 
            Read_Data = self.bar.read(0X7F34000 + (i*4))
            data_read.write(str(Read_Data) + "\n")       
        '''

        #################################################
        #                Layer 7 Start                  #
        #################################################
        # check Layer7 IRQ
        input_file_name = "/proc/interrupts"
        output_file_name_1 = "interrupt.txt"
        output_file_name_2 = "interrupt_old.txt"
        irq_val=0
        while irq_val == 0:                        
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
            input_file.close()
            output_file.close()              

            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            print("layer7 interrupt: 1")
                            irq_val = 1
                            self.L1_IRQ_canvas.itemconfig(self.L7_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)

                    if irq_val != 1:
                        print("layer7 interrupt: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    print("Done")
                    file1.close()
                    file2.close()

            print("extract xdma line Done!\n")

        # Layer 7
        # Read DDR & Conver Format # 512MB
        print("Read DDR")
        Layer7_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X6E58000, End_Address=0X6E8C000)
        Layer7_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer7_1st_Iter_Image1_CH0))     

        Layer7_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X6E8C000, End_Address=0X6EC0000)
        Layer7_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer7_1st_Iter_Image2_CH0))
       
        Layer7_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X6EC0000, End_Address=0X6EF4000)
        Layer7_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer7_1st_Iter_Image3_CH0))

        Layer7_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X6EF4000, End_Address=0X6F28000)
        Layer7_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer7_1st_Iter_Image4_CH0))

        Layer7_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X6F28000, End_Address=0X6F5C000)
        Layer7_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer7_1st_Iter_Image5_CH0))

        Layer7_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X6F5C000, End_Address=0X6F90000)
        Layer7_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer7_1st_Iter_Image6_CH0))

        Layer7_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X6F90000, End_Address=0X6FC4000)
        Layer7_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer7_1st_Iter_Image7_CH0))

        Layer7_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X6FC4000, End_Address=0X6FF8000)
        Layer7_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer7_1st_Iter_Image8_CH0))


        Layer7_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X16E58000, End_Address=0X16E8C000)
        Layer7_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image1_CH1))
        print("ch1 image 1 : ", len(Layer7_1st_Iter_Image1_CH1))

        Layer7_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X16E8C000, End_Address=0X16EC0000)
        Layer7_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer7_1st_Iter_Image2_CH1))

        Layer7_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X16EC0000, End_Address=0X16EF4000)
        Layer7_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer7_1st_Iter_Image3_CH1))

        Layer7_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X16EF4000, End_Address=0X16F28000)
        Layer7_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer7_1st_Iter_Image4_CH1))

        Layer7_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X16F28000, End_Address=0X16F5C000)
        Layer7_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer7_1st_Iter_Image5_CH1))

        Layer7_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X16F5C000, End_Address=0X16F90000)
        Layer7_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer7_1st_Iter_Image6_CH1))

        Layer7_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X16F90000, End_Address=0X16FC4000)
        Layer7_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer7_1st_Iter_Image7_CH1))

        Layer7_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X16FC4000, End_Address=0X16FF8000)
        Layer7_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer7_1st_Iter_Image8_CH1))


        test_out = '1st_iter_result/Layer7_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()


        print("Convert Format")
        Output_Image1_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image1_CH0_256, Layer7_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image2_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image2_CH0_256, Layer7_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image3_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image3_CH0_256, Layer7_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image4_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image4_CH0_256, Layer7_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image5_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image5_CH0_256, Layer7_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image6_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image6_CH0_256, Layer7_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image7_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image7_CH0_256, Layer7_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image8_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image8_CH0_256, Layer7_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)

        OutImages_1st_Layer7 = Output_Image1_Layer7_1st_Iter + Output_Image2_Layer7_1st_Iter + Output_Image3_Layer7_1st_Iter + Output_Image4_Layer7_1st_Iter + \
                            Output_Image5_Layer7_1st_Iter + Output_Image6_Layer7_1st_Iter + Output_Image7_Layer7_1st_Iter + Output_Image8_Layer7_1st_Iter    

        OutImage_1st_Layer7 = torch.tensor([float(value) for value in OutImages_1st_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)

        # Mean, Var
        print("Calculate Mean & Var")
        Mean_1st_Layer7, Var_1st_Layer7 = Cal_mean_var.Forward(OutImage_1st_Layer7)
        Mean_1st_Layer7, Var_1st_Layer7 = Mean_Var_Dec2Bfloat(Mean_1st_Layer7, Var_1st_Layer7, Exponent_Bits, Mantissa_Bits)
        Weight_2nd_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], Mean_1st_Layer7, Var_1st_Layer7, Beta_List[7], Iteration="2")
        
        data_read_mean_var = "result/layer7_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer7:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
   
        # Write DDR
        print("Write DDR")
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer7[0]), Wr_Address=0x6AAE00)
        Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer7[1]), Wr_Address=0x106AAE00)
        
        # resume    
        print("Resume Process")
        self.bar.write(0x20, irq_val)
        print(irq_val)
        irq_val = 0
        self.bar.write(0x20, irq_val)
        print(irq_val)

        '''
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]

        data_read = open("result/layer7_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = self.bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer7_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
            Read_Data = self.bar.read(0X6FF8000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer7_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
            Read_Data = self.bar.read(0X16FF8000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer7_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X8274000-0X80D4000)/4) ): 
            Read_Data = self.bar.read(0X80D4000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''    
        
        end = time.time()
        process_time = (end-start)/60
        print(f'Whole Process: {process_time} mn')
        #################################################
        #                Layer 8 Start                  #
        #################################################

        # Post-Processing Pre-Defined Conditions
        #Post_Start_Signal = "1"

        # OutputImage from Hardware
        print("Read DDR")

        # Post Processing
        #if Post_Start_Signal == "1" or Post_Start_Signal == "1".zfill(4) or Post_Start_Signal == "1".zfill(16):  

        # Layer 8
        # Read DDR & Conver Format # 512MB
        
        Layer8_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0X7198000, End_Address=0X719E800)
        Layer8_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image1_CH0))   
        print("ch0 image 1 : ", len(Layer8_1st_Iter_Image1_CH0))     

        Layer8_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0X719E800, End_Address=0X71A5000)
        Layer8_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image2_CH0))
        print("ch0 image 2 : ", len(Layer8_1st_Iter_Image2_CH0))
    
        Layer8_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0X71A5000, End_Address=0X71AB800)
        Layer8_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image3_CH0))
        print("ch0 image 3 : ", len(Layer8_1st_Iter_Image3_CH0))

        Layer8_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0X71AB800, End_Address=0X71B2000)
        Layer8_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image4_CH0))
        print("ch0 image 4 : ", len(Layer8_1st_Iter_Image4_CH0))

        Layer8_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0X71B2000, End_Address=0X71B8800)
        Layer8_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image5_CH0))
        print("ch0 image 5 : ", len(Layer8_1st_Iter_Image5_CH0))

        Layer8_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0X71B8800, End_Address=0X71BF000)
        Layer8_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image6_CH0))
        print("ch0 image 6 : ", len(Layer8_1st_Iter_Image6_CH0))

        Layer8_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0X71BF000, End_Address=0X71C5800)
        Layer8_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image7_CH0))
        print("ch0 image 7 : ", len(Layer8_1st_Iter_Image7_CH0))

        Layer8_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0X71C5800, End_Address=0X71CC000)
        Layer8_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image8_CH0))
        print("ch0 image 8 : ", len(Layer8_1st_Iter_Image8_CH0))


        Layer8_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0X17198000, End_Address=0X1719E800)
        Layer8_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image1_CH1))   
        print("ch1 image 1 : ", len(Layer8_1st_Iter_Image1_CH1))     

        Layer8_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0X1719E800, End_Address=0X171A5000)
        Layer8_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image2_CH1))
        print("ch1 image 2 : ", len(Layer8_1st_Iter_Image2_CH1))
    
        Layer8_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0X171A5000, End_Address=0X171AB800)
        Layer8_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image3_CH1))
        print("ch1 image 3 : ", len(Layer8_1st_Iter_Image3_CH1))

        Layer8_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0X171AB800, End_Address=0X171B2000)
        Layer8_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image4_CH1))
        print("ch1 image 4 : ", len(Layer8_1st_Iter_Image4_CH1))

        Layer8_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0X171B2000, End_Address=0X171B8800)
        Layer8_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image5_CH1))
        print("ch1 image 5 : ", len(Layer8_1st_Iter_Image5_CH1))

        Layer8_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0X171B8800, End_Address=0X171BF000)
        Layer8_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image6_CH1))
        print("ch1 image 6 : ", len(Layer8_1st_Iter_Image6_CH1))

        Layer8_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0X171BF000, End_Address=0X171C5800)
        Layer8_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image7_CH1))
        print("ch1 image 7 : ", len(Layer8_1st_Iter_Image7_CH1))

        Layer8_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0X171C5800, End_Address=0X171CC000)
        Layer8_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image8_CH1))
        print("ch1 image 8 : ", len(Layer8_1st_Iter_Image8_CH1))

        PostProcessing = Post_Processing(Mode=Mode,
                        Brain_Floating_Point=Brain_Floating_Point,
                        Exponent_Bits=Exponent_Bits,
                        Mantissa_Bits=Mantissa_Bits,
                        OutImage1_Data_CH0=Layer8_1st_Iter_Image1_CH0_256,
                        OutImage1_Data_CH1=Layer8_1st_Iter_Image1_CH1_256,
                        OutImage2_Data_CH0=Layer8_1st_Iter_Image2_CH0_256,
                        OutImage2_Data_CH1=Layer8_1st_Iter_Image2_CH1_256,
                        OutImage3_Data_CH0=Layer8_1st_Iter_Image3_CH0_256,
                        OutImage3_Data_CH1=Layer8_1st_Iter_Image3_CH1_256,
                        OutImage4_Data_CH0=Layer8_1st_Iter_Image4_CH0_256,
                        OutImage4_Data_CH1=Layer8_1st_Iter_Image4_CH1_256,
                        OutImage5_Data_CH0=Layer8_1st_Iter_Image5_CH0_256,
                        OutImage5_Data_CH1=Layer8_1st_Iter_Image5_CH1_256,
                        OutImage6_Data_CH0=Layer8_1st_Iter_Image6_CH0_256,
                        OutImage6_Data_CH1=Layer8_1st_Iter_Image6_CH1_256,
                        OutImage7_Data_CH0=Layer8_1st_Iter_Image7_CH0_256,
                        OutImage7_Data_CH1=Layer8_1st_Iter_Image7_CH1_256,
                        OutImage8_Data_CH0=Layer8_1st_Iter_Image8_CH0_256,
                        OutImage8_Data_CH1=Layer8_1st_Iter_Image8_CH1_256
                        )
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[2]

        data_read = open("result/layer8_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X71CC000-0X7198000)/4) ): 
            Read_Data = self.bar.read(0X7198000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer8_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X71CC000-0X7198000)/4) ): 
            Read_Data = self.bar.read(0X17198000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 
    
        if Mode == "Training":
            Loss, Loss_Gradient = PostProcessing.PostProcessing()
            print(Loss)
            print(Loss_Gradient)
            output_file1 = "result/loss.txt"
            with open(output_file1, mode="w") as output_file_1:
                for item in range(int(Loss)):
                    output_file_1.write(str(item) + "\n")
            output_file2 = "result/loss_gradient.txt"
            with open(output_file2, mode="w") as output_file_2:
                for item in (Loss_Gradient):
                    output_file_2.write(str(item) + "\n")        
            output_file_1.close()
            output_file_2.close()        
            # print(f"Loss Calculation Time: {(Loss_Calculation_Time):.2f} s\n")
        if Mode == "Inference":
            Loss, _ = PostProcessing.PostProcessing()
            print(Loss)
            print("\n")
            # Detection and Boundary Boxes

        data_read = open("result/layer8_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X82A8000-0X8274000)/4) ): 
            Read_Data = self.bar.read(0X8274000 + (i*4))
            data_read.write(str(Read_Data) + "\n")   

        # Backpropagation
        if BN_Param_Converted:
            Backward_Const_List = PreProcessing.Backward_Const_Param_Converted_Func()           
            Average_Per_Channel_List = PreProcessing.Average_Per_Channel_Param_Converted_Func()

        if YOLOv2_Hardware_Backward:
            # Weight_Backward_Layer8 for Soft2Hardware
            Weight_Backward_Layer8 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 1024, Weight_List[8], ["0000"]*125, ["0000"]*125)
            print("Weight_Backward_Layer8: " + str(len(Weight_Backward_Layer8[0])))
            print("Weight_Backward_Layer8: " + str(len(Weight_Backward_Layer8[1])))
            
            # Weight_Backward_Layer7 for Soft2Hardware
            Weight_Backward_Layer7 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], Backward_Const_List[7], Average_Per_Channel_List[7])
            print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer7[0])))
            print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer7[1])))
            
            # Weight_Backward_Layer6 for Soft2Hardware
            Weight_Backward_Layer6 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 512, Weight_List[6], Backward_Const_List[6], Average_Per_Channel_List[6])
            print("Weight_Backward_Layer6: " + str(len(Weight_Backward_Layer6[0])))
            print("Weight_Backward_Layer6: " + str(len(Weight_Backward_Layer6[1])))

            # Weight_Backward_Layer5 for Soft2Hardware
            Weight_Backward_Layer5 = Weight_Hardware_Backward_ReOrdering_OtherLayer(512, 256, Weight_List[5], Backward_Const_List[5], Average_Per_Channel_List[5])
            print("Weight_Backward_Layer5: " + str(len(Weight_Backward_Layer5[0])))
            print("Weight_Backward_Layer5: " + str(len(Weight_Backward_Layer5[1])))

            # Weight_Backward_Layer4 for Soft2Hardware
            Weight_Backward_Layer4 = Weight_Hardware_Backward_ReOrdering_OtherLayer(256, 128, Weight_List[4], Backward_Const_List[4], Average_Per_Channel_List[4])
            print("Weight_Backward_Layer4: " + str(len(Weight_Backward_Layer4[0])))
            print("Weight_Backward_Layer4: " + str(len(Weight_Backward_Layer4[1])))

            # Weight_Backward_Layer3 for Soft2Hardware
            Weight_Backward_Layer3 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 64, Weight_List[3], Backward_Const_List[3], Average_Per_Channel_List[3])
            print("Weight_Backward_Layer3: " + str(len(Weight_Backward_Layer3[0])))
            print("Weight_Backward_Layer3: " + str(len(Weight_Backward_Layer3[1])))

            # Weight_Backward_Layer2 for Soft2Hardware
            Weight_Backward_Layer2 = Weight_Hardware_Backward_ReOrdering_OtherLayer(64, 32, Weight_List[2], Backward_Const_List[2], Average_Per_Channel_List[2])
            print("Weight_Backward_Layer2: " + str(len(Weight_Backward_Layer2[0])))
            print("Weight_Backward_Layer2: " + str(len(Weight_Backward_Layer2[1])))

            # Weight_Backward_Layer1 for Soft2Hardware
            Weight_Backward_Layer1 = Weight_Hardware_Backward_ReOrdering_OtherLayer(32, 16, Weight_List[1], Backward_Const_List[1], Average_Per_Channel_List[1])
            print("Weight_Backward_Layer1: " + str(len(Weight_Backward_Layer1[0])))
            print("Weight_Backward_Layer1: " + str(len(Weight_Backward_Layer1[1])))

            # Weight for Soft2Hardware
            Weight_Backward_Layer0 = Weight_Hardware_Backward_ReOrdering_Layer0(16, 16, Weight_List[0], Backward_Const_List[0], Average_Per_Channel_List[0])
            print("Weight_Layer0: " + str(len(Weight_Backward_Layer0[0])))
            print("Weight_Layer0: " + str(len(Weight_Backward_Layer0[1])))
            
            # Total Weight for Backward: 
            Weight_Backward_CH0 = Weight_Backward_Layer0[0] + Weight_Backward_Layer1[0] + Weight_Backward_Layer2[0] + Weight_Backward_Layer3[0] + Weight_Backward_Layer4[0] + \
                            Weight_Backward_Layer5[0] + Weight_Backward_Layer6[0] + Weight_Backward_Layer7[0] + Weight_Backward_Layer8[0]
            Weight_Backward_CH1 = Weight_Backward_Layer0[1] + Weight_Backward_Layer6[1] + Weight_Backward_Layer2[1] + Weight_Backward_Layer3[1] + Weight_Backward_Layer4[1] + \
                            Weight_Backward_Layer5[1] + Weight_Backward_Layer6[1] + Weight_Backward_Layer7[1] + Weight_Backward_Layer8[1]
            
            # Break 256To32 and Flip the Data: 
            Weight_Backward_CH0 = Break_FlipHex_256To32(Weight_Backward_CH0)
            Weight_Backward_CH1 = Break_FlipHex_256To32(Weight_Backward_CH1)
            
            # Write Weight For Backward into DDR
            Write_DDR(Weight_Backward_CH0,Wr_Address=None)
            print("\t --> " + " Write Weight_Backward_CH0 Done!")
            Write_DDR(Weight_Backward_CH1,Wr_Address=None)
            print("\t --> " + " Write Weight_Backward_CH1 Done!")
            
            # Loss Gradient for Soft2Hardware
            Loss_Gradient1_layer8 = Loss_Gradient[0:1]  
            Loss_Gradient2_layer8 = Loss_Gradient[1:2]  
            Loss_Gradient3_layer8 = Loss_Gradient[2:3]  
            Loss_Gradient4_layer8 = Loss_Gradient[3:4]
            Loss_Gradient5_layer8 = Loss_Gradient[4:5]  
            Loss_Gradient6_layer8 = Loss_Gradient[5:6]  
            Loss_Gradient7_layer8 = Loss_Gradient[6:7]  
            Loss_Gradient8_layer8 = Loss_Gradient[7:8]
            # Loss_Grad1:
            Loss_Grad1_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient1_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad1_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad1_layer8)
            # Loss_Grad2:
            Loss_Grad2_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient2_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad2_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad2_layer8) 
            # Loss_Grad3:  
            Loss_Grad3_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient3_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad3_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad3_layer8) 
            # Loss_Grad4:  
            Loss_Grad4_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient4_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad4_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad4_layer8) 
            # Loss_Grad5:
            Loss_Grad5_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient5_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad5_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad5_layer8) 
            # Loss_Grad6:
            Loss_Grad6_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient6_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad6_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad6_layer8) 
            # Loss_Grad7:  
            Loss_Grad7_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient7_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad7_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad7_layer8) 
            # Loss_Grad8:  
            Loss_Grad8_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient8_layer8, Exponent_Bits, Mantissa_Bits)
            Loss_Grad8_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad8_layer8)
            
            # Separate the DDR Channel: 
            Loss_Grad_layer8_CH0 =  Loss_Grad1_layer8[0] + Loss_Grad2_layer8[0] + Loss_Grad3_layer8[0] + Loss_Grad4_layer8[0] + \
                                    Loss_Grad5_layer8[0] + Loss_Grad6_layer8[0] + Loss_Grad7_layer8[0] + Loss_Grad8_layer8[0]
            
            Loss_Grad_layer8_CH1 =  Loss_Grad1_layer8[1] + Loss_Grad2_layer8[1] + Loss_Grad3_layer8[1] + Loss_Grad4_layer8[1] + \
                                    Loss_Grad5_layer8[1] + Loss_Grad6_layer8[1] + Loss_Grad7_layer8[1] + Loss_Grad8_layer8[1]
            
            # Write Loss Gradient to DDR:
            Write_DDR(Loss_Grad_layer8_CH0, Wr_Address=None)
            print("\t --> " + " Write Loss_Grad_layer8_CH0 Done!")
            Write_DDR(Loss_Grad_layer8_CH1, Wr_Address=None)
            print("\t --> " + " Write Loss_Grad_layer8_CH1 Done!") 

            output_file1 = "result/Loss_Grad_layer8_CH0.txt"
            with open(output_file1, mode="w") as output_file_1:
                for item in Loss_Grad_layer8_CH0:
                    output_file_1.write(str(item) + "\n")
            output_file2 = "result/Loss_Grad_layer8_CH1.txt"
            with open(output_file2, mode="w") as output_file_2:
                for item in Loss_Grad_layer8_CH1:
                    output_file_2.write(str(item) + "\n")        
            output_file_1.close()
            output_file_2.close()

        global Weight_Gradient_Layer0, Weight_Gradient_Layer1, Weight_Gradient_Layer2, Weight_Gradient_Layer3, Weight_Gradient_Layer4,\
           Weight_Gradient_Layer5, Weight_Gradient_Layer6, Weight_Gradient_Layer7, Weight_Gradient_Layer8, Bias_Grad
        global Beta_Gradient_Layer0, Beta_Gradient_Layer1, Beta_Gradient_Layer2, Beta_Gradient_Layer3, Beta_Gradient_Layer4, Beta_Gradient_Layer5, Beta_Gradient_Layer6,\
            Beta_Gradient_Layer7
        # Bias Gradient Calculation
        Bias_Grad = torch.sum(Loss_Gradient, dim=(0, 2, 3))     
        #################################################
        #             Backward Layer 8 Start            #
        #################################################
        # Weight Gradient
        Weight_Gradient1_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer8_CH0_256 = process_input_lines(Weight_Gradient1_Layer8_CH0)
        print("Weight_Gradient1_Layer8_CH0 : ", len(Weight_Gradient1_Layer8_CH0))   

        Weight_Gradient2_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer8_CH0_256 = process_input_lines(Weight_Gradient2_Layer8_CH0)
        print("Weight_Gradient2_Layer8_CH0 : ", len(Weight_Gradient2_Layer8_CH0))    

        Weight_Gradient3_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer8_CH0_256 = process_input_lines(Weight_Gradient3_Layer8_CH0)
        print("Weight_Gradient3_Layer8_CH0 : ", len(Weight_Gradient3_Layer8_CH0)) 

        Weight_Gradient4_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer8_CH0_256 = process_input_lines(Weight_Gradient4_Layer8_CH0)
        print("Weight_Gradient4_Layer8_CH0 : ", len(Weight_Gradient4_Layer8_CH0)) 

        Weight_Gradient5_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer8_CH0_256 = process_input_lines(Weight_Gradient5_Layer8_CH0)
        print("Weight_Gradient5_Layer8_CH0 : ", len(Weight_Gradient5_Layer8_CH0)) 

        Weight_Gradient6_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer8_CH0_256 = process_input_lines(Weight_Gradient6_Layer8_CH0)
        print("Weight_Gradient6_Layer8_CH0 : ", len(Weight_Gradient6_Layer8_CH0)) 

        Weight_Gradient7_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer8_CH0_256 = process_input_lines(Weight_Gradient7_Layer8_CH0)
        print("Weight_Gradient7_Layer8_CH0 : ", len(Weight_Gradient7_Layer8_CH0)) 

        Weight_Gradient8_Layer8_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer8_CH0_256 = process_input_lines(Weight_Gradient8_Layer8_CH0)
        print("Weight_Gradient8_Layer8_CH0 : ", len(Weight_Gradient8_Layer8_CH0)) 

        Weight_Gradient1_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer8_CH1_256 = process_input_lines(Weight_Gradient1_Layer8_CH1)
        print("Weight_Gradient1_Layer8_CH1 : ", len(Weight_Gradient1_Layer8_CH1)) 

        Weight_Gradient2_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer8_CH1_256 = process_input_lines(Weight_Gradient2_Layer8_CH1)
        print("Weight_Gradient2_Layer8_CH1 : ", len(Weight_Gradient2_Layer8_CH1)) 

        Weight_Gradient3_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer8_CH1_256 = process_input_lines(Weight_Gradient3_Layer8_CH1)
        print("Weight_Gradient3_Layer8_CH1 : ", len(Weight_Gradient3_Layer8_CH1)) 

        Weight_Gradient4_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer8_CH1_256 = process_input_lines(Weight_Gradient4_Layer8_CH1)
        print("Weight_Gradient4_Layer8_CH1 : ", len(Weight_Gradient4_Layer8_CH1)) 

        Weight_Gradient5_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer8_CH1_256 = process_input_lines(Weight_Gradient5_Layer8_CH1)
        print("Weight_Gradient5_Layer8_CH1 : ", len(Weight_Gradient5_Layer8_CH1)) 

        Weight_Gradient6_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer8_CH1_256 = process_input_lines(Weight_Gradient6_Layer8_CH1)
        print("Weight_Gradient6_Layer8_CH1 : ", len(Weight_Gradient6_Layer8_CH1)) 

        Weight_Gradient7_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer8_CH1_256 = process_input_lines(Weight_Gradient7_Layer8_CH1)
        print("Weight_Gradient7_Layer8_CH1 : ", len(Weight_Gradient7_Layer8_CH1)) 

        Weight_Gradient8_Layer8_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer8_CH1_256 = process_input_lines(Weight_Gradient8_Layer8_CH1)
        print("Weight_Gradient8_Layer8_CH1 : ", len(Weight_Gradient8_Layer8_CH1)) 
    
        Weight_Gradient1_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer8_CH0_256, Weight_Gradient1_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient2_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer8_CH0_256, Weight_Gradient2_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient3_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer8_CH0_256, Weight_Gradient3_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient4_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer8_CH0_256, Weight_Gradient4_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient5_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer8_CH0_256, Weight_Gradient5_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient6_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer8_CH0_256, Weight_Gradient6_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient7_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer8_CH0_256, Weight_Gradient7_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient8_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer8_CH0_256, Weight_Gradient8_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient_Layer8 = [Weight_Gradient1_Layer8, Weight_Gradient2_Layer8, Weight_Gradient3_Layer8, Weight_Gradient4_Layer8, Weight_Gradient5_Layer8, 
                                Weight_Gradient6_Layer8, Weight_Gradient7_Layer8, Weight_Gradient8_Layer8]
        Weight_Gradient_Layer8 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer8)]   
        Weight_Gradient_Layer8 = torch.tensor([float(value) for value in Weight_Gradient_Layer8], dtype=torch.float32).reshape(128, 1024, 1, 1)     


        #################################################
        #             Backward Layer 7 Start            #
        #################################################

         # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer7_CH0 = process_input_lines(Output_Grad1_Layer7_CH0)

        Output_Grad1_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer7_CH1 = process_input_lines(Output_Grad1_Layer7_CH1)

        Output_Grad2_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer7_CH0 = process_input_lines(Output_Grad2_Layer7_CH0)

        Output_Grad2_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer7_CH1 = process_input_lines(Output_Grad2_Layer7_CH1)

        Output_Grad3_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer7_CH0 = process_input_lines(Output_Grad3_Layer7_CH0)

        Output_Grad3_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer7_CH1 = process_input_lines(Output_Grad3_Layer7_CH1)

        Output_Grad4_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer7_CH0 = process_input_lines(Output_Grad4_Layer7_CH0)

        Output_Grad4_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer7_CH1 = process_input_lines(Output_Grad4_Layer7_CH1)

        Output_Grad5_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer7_CH0 = process_input_lines(Output_Grad5_Layer7_CH0)

        Output_Grad5_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer7_CH1 = process_input_lines(Output_Grad5_Layer7_CH1)

        Output_Grad6_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer7_CH1 = process_input_lines(Output_Grad5_Layer7_CH1)

        Output_Grad6_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer7_CH1 = process_input_lines(Output_Grad6_Layer7_CH1)

        Output_Grad7_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer7_CH0 = process_input_lines(Output_Grad7_Layer7_CH0)

        Output_Grad7_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer7_CH1 = process_input_lines(Output_Grad7_Layer7_CH1)

        Output_Grad8_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer7_CH0 = process_input_lines(Output_Grad8_Layer7_CH0)

        Output_Grad8_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer7_CH1 = process_input_lines(Output_Grad8_Layer7_CH1)

        Output_Grad1_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer7_CH0, Output_Grad1_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad2_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer7_CH0, Output_Grad2_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad3_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer7_CH0, Output_Grad3_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad4_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer7_CH0, Output_Grad4_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad5_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer7_CH0, Output_Grad5_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad6_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer7_CH0, Output_Grad6_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad7_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer7_CH0, Output_Grad7_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad8_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer7_CH0, Output_Grad8_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grads_Layer7 = Output_Grad1_Layer7 + Output_Grad2_Layer7 + Output_Grad3_Layer7 + Output_Grad4_Layer7 + \
                                Output_Grad5_Layer7 + Output_Grad6_Layer7 + Output_Grad7_Layer7 + Output_Grad8_Layer7    
        Output_Grad_Layer7 = torch.tensor([float(value) for value in Output_Grads_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer7 = (Output_Grad_Layer7).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer7_CH0_256 = process_input_lines(Weight_Gradient1_Layer7_CH0)
        print("Weight_Gradient1_Layer7_CH0 : ", len(Weight_Gradient1_Layer7_CH0))   

        Weight_Gradient2_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer7_CH0_256 = process_input_lines(Weight_Gradient2_Layer7_CH0)
        print("Weight_Gradient2_Layer7_CH0 : ", len(Weight_Gradient2_Layer7_CH0))    

        Weight_Gradient3_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer7_CH0_256 = process_input_lines(Weight_Gradient3_Layer7_CH0)
        print("Weight_Gradient3_Layer7_CH0 : ", len(Weight_Gradient3_Layer7_CH0)) 

        Weight_Gradient4_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer7_CH0_256 = process_input_lines(Weight_Gradient4_Layer7_CH0)
        print("Weight_Gradient4_Layer7_CH0 : ", len(Weight_Gradient4_Layer7_CH0)) 

        Weight_Gradient5_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer7_CH0_256 = process_input_lines(Weight_Gradient5_Layer7_CH0)
        print("Weight_Gradient5_Layer7_CH0 : ", len(Weight_Gradient5_Layer7_CH0)) 

        Weight_Gradient6_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer7_CH0_256 = process_input_lines(Weight_Gradient6_Layer7_CH0)
        print("Weight_Gradient6_Layer7_CH0 : ", len(Weight_Gradient6_Layer7_CH0)) 

        Weight_Gradient7_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer7_CH0_256 = process_input_lines(Weight_Gradient7_Layer7_CH0)
        print("Weight_Gradient7_Layer7_CH0 : ", len(Weight_Gradient7_Layer7_CH0)) 

        Weight_Gradient8_Layer7_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer7_CH0_256 = process_input_lines(Weight_Gradient8_Layer7_CH0)
        print("Weight_Gradient8_Layer7_CH0 : ", len(Weight_Gradient8_Layer7_CH0)) 

        Weight_Gradient1_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer7_CH1_256 = process_input_lines(Weight_Gradient1_Layer7_CH1)
        print("Weight_Gradient1_Layer7_CH1 : ", len(Weight_Gradient1_Layer7_CH1)) 

        Weight_Gradient2_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer7_CH1_256 = process_input_lines(Weight_Gradient2_Layer7_CH1)
        print("Weight_Gradient2_Layer7_CH1 : ", len(Weight_Gradient2_Layer7_CH1)) 

        Weight_Gradient3_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer7_CH1_256 = process_input_lines(Weight_Gradient3_Layer7_CH1)
        print("Weight_Gradient3_Layer7_CH1 : ", len(Weight_Gradient3_Layer7_CH1)) 

        Weight_Gradient4_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer7_CH1_256 = process_input_lines(Weight_Gradient4_Layer7_CH1)
        print("Weight_Gradient4_Layer7_CH1 : ", len(Weight_Gradient4_Layer7_CH1)) 

        Weight_Gradient5_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer7_CH1_256 = process_input_lines(Weight_Gradient5_Layer7_CH1)
        print("Weight_Gradient5_Layer7_CH1 : ", len(Weight_Gradient5_Layer7_CH1)) 

        Weight_Gradient6_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer7_CH1_256 = process_input_lines(Weight_Gradient6_Layer7_CH1)
        print("Weight_Gradient6_Layer7_CH1 : ", len(Weight_Gradient6_Layer7_CH1)) 

        Weight_Gradient7_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer7_CH1_256 = process_input_lines(Weight_Gradient7_Layer7_CH1)
        print("Weight_Gradient7_Layer7_CH1 : ", len(Weight_Gradient7_Layer7_CH1)) 

        Weight_Gradient8_Layer7_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer7_CH1_256 = process_input_lines(Weight_Gradient8_Layer7_CH1)
        print("Weight_Gradient8_Layer7_CH1 : ", len(Weight_Gradient8_Layer7_CH1)) 
    
        Weight_Gradient1_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer7_CH0_256, Weight_Gradient1_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient2_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer7_CH0_256, Weight_Gradient2_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient3_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer7_CH0_256, Weight_Gradient3_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient4_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer7_CH0_256, Weight_Gradient4_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient5_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer7_CH0_256, Weight_Gradient5_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient6_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer7_CH0_256, Weight_Gradient6_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient7_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer7_CH0_256, Weight_Gradient7_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient8_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer7_CH0_256, Weight_Gradient8_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient_Layer7 = [Weight_Gradient1_Layer7, Weight_Gradient2_Layer7, Weight_Gradient3_Layer7, Weight_Gradient4_Layer7, Weight_Gradient5_Layer7, 
                                Weight_Gradient6_Layer7, Weight_Gradient7_Layer7, Weight_Gradient8_Layer7]
        Weight_Gradient_Layer7 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer7)]   
        Weight_Gradient_Layer7 = torch.tensor([float(value) for value in Weight_Gradient_Layer7], dtype=torch.float32).reshape(1024, 1024, 3, 3)     


        #################################################
        #             Backward Layer 6 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer6_CH0 = process_input_lines(Output_Grad1_Layer6_CH0)

        Output_Grad1_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer6_CH1 = process_input_lines(Output_Grad1_Layer6_CH1)

        Output_Grad2_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer6_CH0 = process_input_lines(Output_Grad2_Layer6_CH0)

        Output_Grad2_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer6_CH1 = process_input_lines(Output_Grad2_Layer6_CH1)

        Output_Grad3_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer6_CH0 = process_input_lines(Output_Grad3_Layer6_CH0)

        Output_Grad3_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer6_CH1 = process_input_lines(Output_Grad3_Layer6_CH1)

        Output_Grad4_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer6_CH0 = process_input_lines(Output_Grad4_Layer6_CH0)

        Output_Grad4_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer6_CH1 = process_input_lines(Output_Grad4_Layer6_CH1)

        Output_Grad5_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer6_CH0 = process_input_lines(Output_Grad5_Layer6_CH0)

        Output_Grad5_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer6_CH1 = process_input_lines(Output_Grad5_Layer6_CH1)

        Output_Grad6_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer6_CH1 = process_input_lines(Output_Grad5_Layer6_CH1)

        Output_Grad6_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer6_CH1 = process_input_lines(Output_Grad6_Layer6_CH1)

        Output_Grad7_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer6_CH0 = process_input_lines(Output_Grad7_Layer6_CH0)

        Output_Grad7_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer6_CH1 = process_input_lines(Output_Grad7_Layer6_CH1)

        Output_Grad8_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer6_CH0 = process_input_lines(Output_Grad8_Layer6_CH0)

        Output_Grad8_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer6_CH1 = process_input_lines(Output_Grad8_Layer6_CH1)

        Output_Grad1_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer6_CH0, Output_Grad1_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad2_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer6_CH0, Output_Grad2_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad3_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer6_CH0, Output_Grad3_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad4_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer6_CH0, Output_Grad4_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad5_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer6_CH0, Output_Grad5_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad6_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer6_CH0, Output_Grad6_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad7_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer6_CH0, Output_Grad7_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad8_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer6_CH0, Output_Grad8_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grads_Layer6 = Output_Grad1_Layer6 + Output_Grad2_Layer6 + Output_Grad3_Layer6 + Output_Grad4_Layer6 + \
                                Output_Grad5_Layer6 + Output_Grad6_Layer6 + Output_Grad7_Layer6 + Output_Grad8_Layer6    
        Output_Grad_Layer6 = torch.tensor([float(value) for value in Output_Grads_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer6 = (Output_Grad_Layer6).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer6_CH0_256 = process_input_lines(Weight_Gradient1_Layer6_CH0)
        print("Weight_Gradient1_Layer6_CH0 : ", len(Weight_Gradient1_Layer6_CH0))   

        Weight_Gradient2_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer6_CH0_256 = process_input_lines(Weight_Gradient2_Layer6_CH0)
        print("Weight_Gradient2_Layer6_CH0 : ", len(Weight_Gradient2_Layer6_CH0))    

        Weight_Gradient3_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer6_CH0_256 = process_input_lines(Weight_Gradient3_Layer6_CH0)
        print("Weight_Gradient3_Layer6_CH0 : ", len(Weight_Gradient3_Layer6_CH0)) 

        Weight_Gradient4_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer6_CH0_256 = process_input_lines(Weight_Gradient4_Layer6_CH0)
        print("Weight_Gradient4_Layer6_CH0 : ", len(Weight_Gradient4_Layer6_CH0)) 

        Weight_Gradient5_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer6_CH0_256 = process_input_lines(Weight_Gradient5_Layer6_CH0)
        print("Weight_Gradient5_Layer6_CH0 : ", len(Weight_Gradient5_Layer6_CH0)) 

        Weight_Gradient6_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer6_CH0_256 = process_input_lines(Weight_Gradient6_Layer6_CH0)
        print("Weight_Gradient6_Layer6_CH0 : ", len(Weight_Gradient6_Layer6_CH0)) 

        Weight_Gradient7_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer6_CH0_256 = process_input_lines(Weight_Gradient7_Layer6_CH0)
        print("Weight_Gradient7_Layer6_CH0 : ", len(Weight_Gradient7_Layer6_CH0)) 

        Weight_Gradient8_Layer6_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer6_CH0_256 = process_input_lines(Weight_Gradient8_Layer6_CH0)
        print("Weight_Gradient8_Layer6_CH0 : ", len(Weight_Gradient8_Layer6_CH0)) 

        Weight_Gradient1_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer6_CH1_256 = process_input_lines(Weight_Gradient1_Layer6_CH1)
        print("Weight_Gradient1_Layer6_CH1 : ", len(Weight_Gradient1_Layer6_CH1)) 

        Weight_Gradient2_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer6_CH1_256 = process_input_lines(Weight_Gradient2_Layer6_CH1)
        print("Weight_Gradient2_Layer6_CH1 : ", len(Weight_Gradient2_Layer6_CH1)) 

        Weight_Gradient3_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer6_CH1_256 = process_input_lines(Weight_Gradient3_Layer6_CH1)
        print("Weight_Gradient3_Layer6_CH1 : ", len(Weight_Gradient3_Layer6_CH1)) 

        Weight_Gradient4_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer6_CH1_256 = process_input_lines(Weight_Gradient4_Layer6_CH1)
        print("Weight_Gradient4_Layer6_CH1 : ", len(Weight_Gradient4_Layer6_CH1)) 

        Weight_Gradient5_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer6_CH1_256 = process_input_lines(Weight_Gradient5_Layer6_CH1)
        print("Weight_Gradient5_Layer6_CH1 : ", len(Weight_Gradient5_Layer6_CH1)) 

        Weight_Gradient6_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer6_CH1_256 = process_input_lines(Weight_Gradient6_Layer6_CH1)
        print("Weight_Gradient6_Layer6_CH1 : ", len(Weight_Gradient6_Layer6_CH1)) 

        Weight_Gradient7_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer6_CH1_256 = process_input_lines(Weight_Gradient7_Layer6_CH1)
        print("Weight_Gradient7_Layer6_CH1 : ", len(Weight_Gradient7_Layer6_CH1)) 

        Weight_Gradient8_Layer6_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer6_CH1_256 = process_input_lines(Weight_Gradient8_Layer6_CH1)
        print("Weight_Gradient8_Layer6_CH1 : ", len(Weight_Gradient8_Layer6_CH1)) 
    
        Weight_Gradient1_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer6_CH0_256, Weight_Gradient1_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient2_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer6_CH0_256, Weight_Gradient2_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient3_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer6_CH0_256, Weight_Gradient3_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient4_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer6_CH0_256, Weight_Gradient4_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient5_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer6_CH0_256, Weight_Gradient5_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient6_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer6_CH0_256, Weight_Gradient6_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient7_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer6_CH0_256, Weight_Gradient7_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient8_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer6_CH0_256, Weight_Gradient8_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient_Layer6 = [Weight_Gradient1_Layer6, Weight_Gradient2_Layer6, Weight_Gradient3_Layer6, Weight_Gradient4_Layer6, Weight_Gradient5_Layer6, 
                                Weight_Gradient6_Layer6, Weight_Gradient7_Layer6, Weight_Gradient8_Layer6]
        Weight_Gradient_Layer6 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer6)]   
        Weight_Gradient_Layer6 = torch.tensor([float(value) for value in Weight_Gradient_Layer6], dtype=torch.float32).reshape(1024, 512, 3, 3)    

        #################################################
        #             Backward Layer 5 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer5_CH0 = process_input_lines(Output_Grad1_Layer5_CH0)

        Output_Grad1_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer5_CH1 = process_input_lines(Output_Grad1_Layer5_CH1)

        Output_Grad2_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer5_CH0 = process_input_lines(Output_Grad2_Layer5_CH0)

        Output_Grad2_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer5_CH1 = process_input_lines(Output_Grad2_Layer5_CH1)

        Output_Grad3_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer5_CH0 = process_input_lines(Output_Grad3_Layer5_CH0)

        Output_Grad3_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer5_CH1 = process_input_lines(Output_Grad3_Layer5_CH1)

        Output_Grad4_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer5_CH0 = process_input_lines(Output_Grad4_Layer5_CH0)

        Output_Grad4_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer5_CH1 = process_input_lines(Output_Grad4_Layer5_CH1)

        Output_Grad5_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer5_CH0 = process_input_lines(Output_Grad5_Layer5_CH0)

        Output_Grad5_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer5_CH1 = process_input_lines(Output_Grad5_Layer5_CH1)

        Output_Grad6_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer5_CH1 = process_input_lines(Output_Grad5_Layer5_CH1)

        Output_Grad6_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer5_CH1 = process_input_lines(Output_Grad6_Layer5_CH1)

        Output_Grad7_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer5_CH0 = process_input_lines(Output_Grad7_Layer5_CH0)

        Output_Grad7_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer5_CH1 = process_input_lines(Output_Grad7_Layer5_CH1)

        Output_Grad8_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer5_CH0 = process_input_lines(Output_Grad8_Layer5_CH0)

        Output_Grad8_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer5_CH1 = process_input_lines(Output_Grad8_Layer5_CH1)

        Output_Grad1_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer5_CH0, Output_Grad1_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad2_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer5_CH0, Output_Grad2_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad3_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer5_CH0, Output_Grad3_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad4_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer5_CH0, Output_Grad4_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad5_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer5_CH0, Output_Grad5_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad6_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer5_CH0, Output_Grad6_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad7_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer5_CH0, Output_Grad7_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad8_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer5_CH0, Output_Grad8_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grads_Layer5 = Output_Grad1_Layer5 + Output_Grad2_Layer5 + Output_Grad3_Layer5 + Output_Grad4_Layer5 + \
                                Output_Grad5_Layer5 + Output_Grad6_Layer5 + Output_Grad7_Layer5 + Output_Grad8_Layer5    
        Output_Grad_Layer5 = torch.tensor([float(value) for value in Output_Grads_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer5 = (Output_Grad_Layer5).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer5_CH0_256 = process_input_lines(Weight_Gradient1_Layer5_CH0)
        print("Weight_Gradient1_Layer5_CH0 : ", len(Weight_Gradient1_Layer5_CH0))   

        Weight_Gradient2_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer5_CH0_256 = process_input_lines(Weight_Gradient2_Layer5_CH0)
        print("Weight_Gradient2_Layer5_CH0 : ", len(Weight_Gradient2_Layer5_CH0))    

        Weight_Gradient3_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer5_CH0_256 = process_input_lines(Weight_Gradient3_Layer5_CH0)
        print("Weight_Gradient3_Layer5_CH0 : ", len(Weight_Gradient3_Layer5_CH0)) 

        Weight_Gradient4_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer5_CH0_256 = process_input_lines(Weight_Gradient4_Layer5_CH0)
        print("Weight_Gradient4_Layer5_CH0 : ", len(Weight_Gradient4_Layer5_CH0)) 

        Weight_Gradient5_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer5_CH0_256 = process_input_lines(Weight_Gradient5_Layer5_CH0)
        print("Weight_Gradient5_Layer5_CH0 : ", len(Weight_Gradient5_Layer5_CH0)) 

        Weight_Gradient6_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer5_CH0_256 = process_input_lines(Weight_Gradient6_Layer5_CH0)
        print("Weight_Gradient6_Layer5_CH0 : ", len(Weight_Gradient6_Layer5_CH0)) 

        Weight_Gradient7_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer5_CH0_256 = process_input_lines(Weight_Gradient7_Layer5_CH0)
        print("Weight_Gradient7_Layer5_CH0 : ", len(Weight_Gradient7_Layer5_CH0)) 

        Weight_Gradient8_Layer5_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer5_CH0_256 = process_input_lines(Weight_Gradient8_Layer5_CH0)
        print("Weight_Gradient8_Layer5_CH0 : ", len(Weight_Gradient8_Layer5_CH0)) 

        Weight_Gradient1_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer5_CH1_256 = process_input_lines(Weight_Gradient1_Layer5_CH1)
        print("Weight_Gradient1_Layer5_CH1 : ", len(Weight_Gradient1_Layer5_CH1)) 

        Weight_Gradient2_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer5_CH1_256 = process_input_lines(Weight_Gradient2_Layer5_CH1)
        print("Weight_Gradient2_Layer5_CH1 : ", len(Weight_Gradient2_Layer5_CH1)) 

        Weight_Gradient3_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer5_CH1_256 = process_input_lines(Weight_Gradient3_Layer5_CH1)
        print("Weight_Gradient3_Layer5_CH1 : ", len(Weight_Gradient3_Layer5_CH1)) 

        Weight_Gradient4_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer5_CH1_256 = process_input_lines(Weight_Gradient4_Layer5_CH1)
        print("Weight_Gradient4_Layer5_CH1 : ", len(Weight_Gradient4_Layer5_CH1)) 

        Weight_Gradient5_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer5_CH1_256 = process_input_lines(Weight_Gradient5_Layer5_CH1)
        print("Weight_Gradient5_Layer5_CH1 : ", len(Weight_Gradient5_Layer5_CH1)) 

        Weight_Gradient6_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer5_CH1_256 = process_input_lines(Weight_Gradient6_Layer5_CH1)
        print("Weight_Gradient6_Layer5_CH1 : ", len(Weight_Gradient6_Layer5_CH1)) 

        Weight_Gradient7_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer5_CH1_256 = process_input_lines(Weight_Gradient7_Layer5_CH1)
        print("Weight_Gradient7_Layer5_CH1 : ", len(Weight_Gradient7_Layer5_CH1)) 

        Weight_Gradient8_Layer5_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer5_CH1_256 = process_input_lines(Weight_Gradient8_Layer5_CH1)
        print("Weight_Gradient8_Layer5_CH1 : ", len(Weight_Gradient8_Layer5_CH1)) 
    
        Weight_Gradient1_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer5_CH0_256, Weight_Gradient1_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient2_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer5_CH0_256, Weight_Gradient2_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient3_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer5_CH0_256, Weight_Gradient3_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient4_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer5_CH0_256, Weight_Gradient4_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient5_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer5_CH0_256, Weight_Gradient5_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient6_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer5_CH0_256, Weight_Gradient6_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient7_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer5_CH0_256, Weight_Gradient7_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient8_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer5_CH0_256, Weight_Gradient8_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient_Layer5 = [Weight_Gradient1_Layer5, Weight_Gradient2_Layer5, Weight_Gradient3_Layer5, Weight_Gradient4_Layer5, Weight_Gradient5_Layer5, 
                                Weight_Gradient6_Layer5, Weight_Gradient7_Layer5, Weight_Gradient8_Layer5]
        Weight_Gradient_Layer5 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer5)]   
        Weight_Gradient_Layer5 = torch.tensor([float(value) for value in Weight_Gradient_Layer5], dtype=torch.float32).reshape(512, 256, 3, 3)      

        #################################################
        #             Backward Layer 4 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer4_CH0 = process_input_lines(Output_Grad1_Layer4_CH0)

        Output_Grad1_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer4_CH1 = process_input_lines(Output_Grad1_Layer4_CH1)

        Output_Grad2_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer4_CH0 = process_input_lines(Output_Grad2_Layer4_CH0)

        Output_Grad2_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer4_CH1 = process_input_lines(Output_Grad2_Layer4_CH1)

        Output_Grad3_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer4_CH0 = process_input_lines(Output_Grad3_Layer4_CH0)

        Output_Grad3_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer4_CH1 = process_input_lines(Output_Grad3_Layer4_CH1)

        Output_Grad4_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer4_CH0 = process_input_lines(Output_Grad4_Layer4_CH0)

        Output_Grad4_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer4_CH1 = process_input_lines(Output_Grad4_Layer4_CH1)

        Output_Grad5_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer4_CH0 = process_input_lines(Output_Grad5_Layer4_CH0)

        Output_Grad5_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer4_CH1 = process_input_lines(Output_Grad5_Layer4_CH1)

        Output_Grad6_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer4_CH1 = process_input_lines(Output_Grad5_Layer4_CH1)

        Output_Grad6_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer4_CH1 = process_input_lines(Output_Grad6_Layer4_CH1)

        Output_Grad7_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer4_CH0 = process_input_lines(Output_Grad7_Layer4_CH0)

        Output_Grad7_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer4_CH1 = process_input_lines(Output_Grad7_Layer4_CH1)

        Output_Grad8_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer4_CH0 = process_input_lines(Output_Grad8_Layer4_CH0)

        Output_Grad8_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer4_CH1 = process_input_lines(Output_Grad8_Layer4_CH1)

        Output_Grad1_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer4_CH0, Output_Grad1_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad2_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer4_CH0, Output_Grad2_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad3_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer4_CH0, Output_Grad3_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad4_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer4_CH0, Output_Grad4_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad5_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer4_CH0, Output_Grad5_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad6_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer4_CH0, Output_Grad6_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad7_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer4_CH0, Output_Grad7_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grad8_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer4_CH0, Output_Grad8_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Grads_Layer4 = Output_Grad1_Layer4 + Output_Grad2_Layer4 + Output_Grad3_Layer4 + Output_Grad4_Layer4 + \
                                Output_Grad5_Layer4 + Output_Grad6_Layer4 + Output_Grad7_Layer4 + Output_Grad8_Layer4    
        Output_Grad_Layer4 = torch.tensor([float(value) for value in Output_Grads_Layer4], dtype=torch.float32).reshape(8, 256, 26, 26)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer4 = (Output_Grad_Layer4).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer4_CH0_256 = process_input_lines(Weight_Gradient1_Layer4_CH0)
        print("Weight_Gradient1_Layer4_CH0 : ", len(Weight_Gradient1_Layer4_CH0))   

        Weight_Gradient2_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer4_CH0_256 = process_input_lines(Weight_Gradient2_Layer4_CH0)
        print("Weight_Gradient2_Layer4_CH0 : ", len(Weight_Gradient2_Layer4_CH0))    

        Weight_Gradient3_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer4_CH0_256 = process_input_lines(Weight_Gradient3_Layer4_CH0)
        print("Weight_Gradient3_Layer4_CH0 : ", len(Weight_Gradient3_Layer4_CH0)) 

        Weight_Gradient4_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer4_CH0_256 = process_input_lines(Weight_Gradient4_Layer4_CH0)
        print("Weight_Gradient4_Layer4_CH0 : ", len(Weight_Gradient4_Layer4_CH0)) 

        Weight_Gradient5_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer4_CH0_256 = process_input_lines(Weight_Gradient5_Layer4_CH0)
        print("Weight_Gradient5_Layer4_CH0 : ", len(Weight_Gradient5_Layer4_CH0)) 

        Weight_Gradient6_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer4_CH0_256 = process_input_lines(Weight_Gradient6_Layer4_CH0)
        print("Weight_Gradient6_Layer4_CH0 : ", len(Weight_Gradient6_Layer4_CH0)) 

        Weight_Gradient7_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer4_CH0_256 = process_input_lines(Weight_Gradient7_Layer4_CH0)
        print("Weight_Gradient7_Layer4_CH0 : ", len(Weight_Gradient7_Layer4_CH0)) 

        Weight_Gradient8_Layer4_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer4_CH0_256 = process_input_lines(Weight_Gradient8_Layer4_CH0)
        print("Weight_Gradient8_Layer4_CH0 : ", len(Weight_Gradient8_Layer4_CH0)) 

        Weight_Gradient1_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer4_CH1_256 = process_input_lines(Weight_Gradient1_Layer4_CH1)
        print("Weight_Gradient1_Layer4_CH1 : ", len(Weight_Gradient1_Layer4_CH1)) 

        Weight_Gradient2_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer4_CH1_256 = process_input_lines(Weight_Gradient2_Layer4_CH1)
        print("Weight_Gradient2_Layer4_CH1 : ", len(Weight_Gradient2_Layer4_CH1)) 

        Weight_Gradient3_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer4_CH1_256 = process_input_lines(Weight_Gradient3_Layer4_CH1)
        print("Weight_Gradient3_Layer4_CH1 : ", len(Weight_Gradient3_Layer4_CH1)) 

        Weight_Gradient4_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer4_CH1_256 = process_input_lines(Weight_Gradient4_Layer4_CH1)
        print("Weight_Gradient4_Layer4_CH1 : ", len(Weight_Gradient4_Layer4_CH1)) 

        Weight_Gradient5_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer4_CH1_256 = process_input_lines(Weight_Gradient5_Layer4_CH1)
        print("Weight_Gradient5_Layer4_CH1 : ", len(Weight_Gradient5_Layer4_CH1)) 

        Weight_Gradient6_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer4_CH1_256 = process_input_lines(Weight_Gradient6_Layer4_CH1)
        print("Weight_Gradient6_Layer4_CH1 : ", len(Weight_Gradient6_Layer4_CH1)) 

        Weight_Gradient7_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer4_CH1_256 = process_input_lines(Weight_Gradient7_Layer4_CH1)
        print("Weight_Gradient7_Layer4_CH1 : ", len(Weight_Gradient7_Layer4_CH1)) 

        Weight_Gradient8_Layer4_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer4_CH1_256 = process_input_lines(Weight_Gradient8_Layer4_CH1)
        print("Weight_Gradient8_Layer4_CH1 : ", len(Weight_Gradient8_Layer4_CH1)) 
    
        Weight_Gradient1_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer4_CH0_256, Weight_Gradient1_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient2_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer4_CH0_256, Weight_Gradient2_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient3_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer4_CH0_256, Weight_Gradient3_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient4_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer4_CH0_256, Weight_Gradient4_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient5_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer4_CH0_256, Weight_Gradient5_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient6_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer4_CH0_256, Weight_Gradient6_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient7_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer4_CH0_256, Weight_Gradient7_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient8_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer4_CH0_256, Weight_Gradient8_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient_Layer4 = [Weight_Gradient1_Layer4, Weight_Gradient2_Layer4, Weight_Gradient3_Layer4, Weight_Gradient4_Layer4, Weight_Gradient5_Layer4, 
                                Weight_Gradient6_Layer4, Weight_Gradient7_Layer4, Weight_Gradient8_Layer4]
        Weight_Gradient_Layer4 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer4)]   
        Weight_Gradient_Layer4 = torch.tensor([float(value) for value in Weight_Gradient_Layer4], dtype=torch.float32).reshape(256, 128, 3, 3)   

        #################################################
        #             Backward Layer 3 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer3_CH0 = process_input_lines(Output_Grad1_Layer3_CH0)

        Output_Grad1_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer3_CH1 = process_input_lines(Output_Grad1_Layer3_CH1)

        Output_Grad2_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer3_CH0 = process_input_lines(Output_Grad2_Layer3_CH0)

        Output_Grad2_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer3_CH1 = process_input_lines(Output_Grad2_Layer3_CH1)

        Output_Grad3_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer3_CH0 = process_input_lines(Output_Grad3_Layer3_CH0)

        Output_Grad3_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer3_CH1 = process_input_lines(Output_Grad3_Layer3_CH1)

        Output_Grad4_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer3_CH0 = process_input_lines(Output_Grad4_Layer3_CH0)

        Output_Grad4_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer3_CH1 = process_input_lines(Output_Grad4_Layer3_CH1)

        Output_Grad5_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer3_CH0 = process_input_lines(Output_Grad5_Layer3_CH0)

        Output_Grad5_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer3_CH1 = process_input_lines(Output_Grad5_Layer3_CH1)

        Output_Grad6_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer3_CH1 = process_input_lines(Output_Grad5_Layer3_CH1)

        Output_Grad6_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer3_CH1 = process_input_lines(Output_Grad6_Layer3_CH1)

        Output_Grad7_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer3_CH0 = process_input_lines(Output_Grad7_Layer3_CH0)

        Output_Grad7_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer3_CH1 = process_input_lines(Output_Grad7_Layer3_CH1)

        Output_Grad8_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer3_CH0 = process_input_lines(Output_Grad8_Layer3_CH0)

        Output_Grad8_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer3_CH1 = process_input_lines(Output_Grad8_Layer3_CH1)

        Output_Grad1_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer3_CH0, Output_Grad1_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad2_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer3_CH0, Output_Grad2_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad3_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer3_CH0, Output_Grad3_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad4_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer3_CH0, Output_Grad4_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad5_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer3_CH0, Output_Grad5_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad6_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer3_CH0, Output_Grad6_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad7_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer3_CH0, Output_Grad7_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grad8_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer3_CH0, Output_Grad8_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Grads_Layer3 = Output_Grad1_Layer3 + Output_Grad2_Layer3 + Output_Grad3_Layer3 + Output_Grad4_Layer3 + \
                                Output_Grad5_Layer3 + Output_Grad6_Layer3 + Output_Grad7_Layer3 + Output_Grad8_Layer3    
        Output_Grad_Layer3 = torch.tensor([float(value) for value in Output_Grads_Layer3], dtype=torch.float32).reshape(8, 128, 52, 52)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer3 = (Output_Grad_Layer3).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer3_CH0_256 = process_input_lines(Weight_Gradient1_Layer3_CH0)
        print("Weight_Gradient1_Layer3_CH0 : ", len(Weight_Gradient1_Layer3_CH0))   

        Weight_Gradient2_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer3_CH0_256 = process_input_lines(Weight_Gradient2_Layer3_CH0)
        print("Weight_Gradient2_Layer3_CH0 : ", len(Weight_Gradient2_Layer3_CH0))    

        Weight_Gradient3_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer3_CH0_256 = process_input_lines(Weight_Gradient3_Layer3_CH0)
        print("Weight_Gradient3_Layer3_CH0 : ", len(Weight_Gradient3_Layer3_CH0)) 

        Weight_Gradient4_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer3_CH0_256 = process_input_lines(Weight_Gradient4_Layer3_CH0)
        print("Weight_Gradient4_Layer3_CH0 : ", len(Weight_Gradient4_Layer3_CH0)) 

        Weight_Gradient5_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer3_CH0_256 = process_input_lines(Weight_Gradient5_Layer3_CH0)
        print("Weight_Gradient5_Layer3_CH0 : ", len(Weight_Gradient5_Layer3_CH0)) 

        Weight_Gradient6_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer3_CH0_256 = process_input_lines(Weight_Gradient6_Layer3_CH0)
        print("Weight_Gradient6_Layer3_CH0 : ", len(Weight_Gradient6_Layer3_CH0)) 

        Weight_Gradient7_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer3_CH0_256 = process_input_lines(Weight_Gradient7_Layer3_CH0)
        print("Weight_Gradient7_Layer3_CH0 : ", len(Weight_Gradient7_Layer3_CH0)) 

        Weight_Gradient8_Layer3_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer3_CH0_256 = process_input_lines(Weight_Gradient8_Layer3_CH0)
        print("Weight_Gradient8_Layer3_CH0 : ", len(Weight_Gradient8_Layer3_CH0)) 

        Weight_Gradient1_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer3_CH1_256 = process_input_lines(Weight_Gradient1_Layer3_CH1)
        print("Weight_Gradient1_Layer3_CH1 : ", len(Weight_Gradient1_Layer3_CH1)) 

        Weight_Gradient2_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer3_CH1_256 = process_input_lines(Weight_Gradient2_Layer3_CH1)
        print("Weight_Gradient2_Layer3_CH1 : ", len(Weight_Gradient2_Layer3_CH1)) 

        Weight_Gradient3_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer3_CH1_256 = process_input_lines(Weight_Gradient3_Layer3_CH1)
        print("Weight_Gradient3_Layer3_CH1 : ", len(Weight_Gradient3_Layer3_CH1)) 

        Weight_Gradient4_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer3_CH1_256 = process_input_lines(Weight_Gradient4_Layer3_CH1)
        print("Weight_Gradient4_Layer3_CH1 : ", len(Weight_Gradient4_Layer3_CH1)) 

        Weight_Gradient5_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer3_CH1_256 = process_input_lines(Weight_Gradient5_Layer3_CH1)
        print("Weight_Gradient5_Layer3_CH1 : ", len(Weight_Gradient5_Layer3_CH1)) 

        Weight_Gradient6_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer3_CH1_256 = process_input_lines(Weight_Gradient6_Layer3_CH1)
        print("Weight_Gradient6_Layer3_CH1 : ", len(Weight_Gradient6_Layer3_CH1)) 

        Weight_Gradient7_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer3_CH1_256 = process_input_lines(Weight_Gradient7_Layer3_CH1)
        print("Weight_Gradient7_Layer3_CH1 : ", len(Weight_Gradient7_Layer3_CH1)) 

        Weight_Gradient8_Layer3_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer3_CH1_256 = process_input_lines(Weight_Gradient8_Layer3_CH1)
        print("Weight_Gradient8_Layer3_CH1 : ", len(Weight_Gradient8_Layer3_CH1)) 
    
        Weight_Gradient1_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer3_CH0_256, Weight_Gradient1_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient2_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer3_CH0_256, Weight_Gradient2_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient3_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer3_CH0_256, Weight_Gradient3_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient4_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer3_CH0_256, Weight_Gradient4_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient5_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer3_CH0_256, Weight_Gradient5_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient6_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer3_CH0_256, Weight_Gradient6_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient7_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer3_CH0_256, Weight_Gradient7_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient8_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer3_CH0_256, Weight_Gradient8_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient_Layer3 = [Weight_Gradient1_Layer3, Weight_Gradient2_Layer3, Weight_Gradient3_Layer3, Weight_Gradient4_Layer3, Weight_Gradient5_Layer3, 
                                Weight_Gradient6_Layer3, Weight_Gradient7_Layer3, Weight_Gradient8_Layer3]
        Weight_Gradient_Layer3 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer3)]   
        Weight_Gradient_Layer3 = torch.tensor([float(value) for value in Weight_Gradient_Layer3], dtype=torch.float32).reshape(128, 64, 3, 3)   

        #################################################
        #             Backward Layer 2 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer2_CH0 = process_input_lines(Output_Grad1_Layer2_CH0)

        Output_Grad1_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer2_CH1 = process_input_lines(Output_Grad1_Layer2_CH1)

        Output_Grad2_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer2_CH0 = process_input_lines(Output_Grad2_Layer2_CH0)

        Output_Grad2_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer2_CH1 = process_input_lines(Output_Grad2_Layer2_CH1)

        Output_Grad3_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer2_CH0 = process_input_lines(Output_Grad3_Layer2_CH0)

        Output_Grad3_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer2_CH1 = process_input_lines(Output_Grad3_Layer2_CH1)

        Output_Grad4_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer2_CH0 = process_input_lines(Output_Grad4_Layer2_CH0)

        Output_Grad4_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer2_CH1 = process_input_lines(Output_Grad4_Layer2_CH1)

        Output_Grad5_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer2_CH0 = process_input_lines(Output_Grad5_Layer2_CH0)

        Output_Grad5_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer2_CH1 = process_input_lines(Output_Grad5_Layer2_CH1)

        Output_Grad6_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer2_CH1 = process_input_lines(Output_Grad5_Layer2_CH1)

        Output_Grad6_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer2_CH1 = process_input_lines(Output_Grad6_Layer2_CH1)

        Output_Grad7_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer2_CH0 = process_input_lines(Output_Grad7_Layer2_CH0)

        Output_Grad7_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer2_CH1 = process_input_lines(Output_Grad7_Layer2_CH1)

        Output_Grad8_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer2_CH0 = process_input_lines(Output_Grad8_Layer2_CH0)

        Output_Grad8_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer2_CH1 = process_input_lines(Output_Grad8_Layer2_CH1)

        Output_Grad1_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer2_CH0, Output_Grad1_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad2_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer2_CH0, Output_Grad2_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad3_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer2_CH0, Output_Grad3_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad4_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer2_CH0, Output_Grad4_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad5_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer2_CH0, Output_Grad5_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad6_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer2_CH0, Output_Grad6_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad7_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer2_CH0, Output_Grad7_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grad8_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer2_CH0, Output_Grad8_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Grads_Layer2 = Output_Grad1_Layer2 + Output_Grad2_Layer2 + Output_Grad3_Layer2 + Output_Grad4_Layer2 + \
                                Output_Grad5_Layer2 + Output_Grad6_Layer2 + Output_Grad7_Layer2 + Output_Grad8_Layer2    
        Output_Grad_Layer2 = torch.tensor([float(value) for value in Output_Grads_Layer2], dtype=torch.float32).reshape(8, 64, 104, 104)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer2 = (Output_Grad_Layer2).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer2_CH0_256 = process_input_lines(Weight_Gradient1_Layer2_CH0)
        print("Weight_Gradient1_Layer2_CH0 : ", len(Weight_Gradient1_Layer2_CH0))   

        Weight_Gradient2_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer2_CH0_256 = process_input_lines(Weight_Gradient2_Layer2_CH0)
        print("Weight_Gradient2_Layer2_CH0 : ", len(Weight_Gradient2_Layer2_CH0))    

        Weight_Gradient3_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer2_CH0_256 = process_input_lines(Weight_Gradient3_Layer2_CH0)
        print("Weight_Gradient3_Layer2_CH0 : ", len(Weight_Gradient3_Layer2_CH0)) 

        Weight_Gradient4_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer2_CH0_256 = process_input_lines(Weight_Gradient4_Layer2_CH0)
        print("Weight_Gradient4_Layer2_CH0 : ", len(Weight_Gradient4_Layer2_CH0)) 

        Weight_Gradient5_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer2_CH0_256 = process_input_lines(Weight_Gradient5_Layer2_CH0)
        print("Weight_Gradient5_Layer2_CH0 : ", len(Weight_Gradient5_Layer2_CH0)) 

        Weight_Gradient6_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer2_CH0_256 = process_input_lines(Weight_Gradient6_Layer2_CH0)
        print("Weight_Gradient6_Layer2_CH0 : ", len(Weight_Gradient6_Layer2_CH0)) 

        Weight_Gradient7_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer2_CH0_256 = process_input_lines(Weight_Gradient7_Layer2_CH0)
        print("Weight_Gradient7_Layer2_CH0 : ", len(Weight_Gradient7_Layer2_CH0)) 

        Weight_Gradient8_Layer2_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer2_CH0_256 = process_input_lines(Weight_Gradient8_Layer2_CH0)
        print("Weight_Gradient8_Layer2_CH0 : ", len(Weight_Gradient8_Layer2_CH0)) 

        Weight_Gradient1_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer2_CH1_256 = process_input_lines(Weight_Gradient1_Layer2_CH1)
        print("Weight_Gradient1_Layer2_CH1 : ", len(Weight_Gradient1_Layer2_CH1)) 

        Weight_Gradient2_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer2_CH1_256 = process_input_lines(Weight_Gradient2_Layer2_CH1)
        print("Weight_Gradient2_Layer2_CH1 : ", len(Weight_Gradient2_Layer2_CH1)) 

        Weight_Gradient3_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer2_CH1_256 = process_input_lines(Weight_Gradient3_Layer2_CH1)
        print("Weight_Gradient3_Layer2_CH1 : ", len(Weight_Gradient3_Layer2_CH1)) 

        Weight_Gradient4_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer2_CH1_256 = process_input_lines(Weight_Gradient4_Layer2_CH1)
        print("Weight_Gradient4_Layer2_CH1 : ", len(Weight_Gradient4_Layer2_CH1)) 

        Weight_Gradient5_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer2_CH1_256 = process_input_lines(Weight_Gradient5_Layer2_CH1)
        print("Weight_Gradient5_Layer2_CH1 : ", len(Weight_Gradient5_Layer2_CH1)) 

        Weight_Gradient6_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer2_CH1_256 = process_input_lines(Weight_Gradient6_Layer2_CH1)
        print("Weight_Gradient6_Layer2_CH1 : ", len(Weight_Gradient6_Layer2_CH1)) 

        Weight_Gradient7_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer2_CH1_256 = process_input_lines(Weight_Gradient7_Layer2_CH1)
        print("Weight_Gradient7_Layer2_CH1 : ", len(Weight_Gradient7_Layer2_CH1)) 

        Weight_Gradient8_Layer2_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer2_CH1_256 = process_input_lines(Weight_Gradient8_Layer2_CH1)
        print("Weight_Gradient8_Layer2_CH1 : ", len(Weight_Gradient8_Layer2_CH1)) 
    
        Weight_Gradient1_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer2_CH0_256, Weight_Gradient1_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient2_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer2_CH0_256, Weight_Gradient2_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient3_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer2_CH0_256, Weight_Gradient3_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient4_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer2_CH0_256, Weight_Gradient4_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient5_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer2_CH0_256, Weight_Gradient5_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient6_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer2_CH0_256, Weight_Gradient6_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient7_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer2_CH0_256, Weight_Gradient7_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient8_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer2_CH0_256, Weight_Gradient8_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient_Layer2 = [Weight_Gradient1_Layer2, Weight_Gradient2_Layer2, Weight_Gradient3_Layer2, Weight_Gradient4_Layer2, Weight_Gradient5_Layer2, 
                                Weight_Gradient6_Layer2, Weight_Gradient7_Layer2, Weight_Gradient8_Layer2]
        Weight_Gradient_Layer2 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer2)]   
        Weight_Gradient_Layer2 = torch.tensor([float(value) for value in Weight_Gradient_Layer2], dtype=torch.float32).reshape(64, 32, 3, 3)   

        #################################################
        #             Backward Layer 1 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer1_CH0 = process_input_lines(Output_Grad1_Layer1_CH0)

        Output_Grad1_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer1_CH1 = process_input_lines(Output_Grad1_Layer1_CH1)

        Output_Grad2_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer1_CH0 = process_input_lines(Output_Grad2_Layer1_CH0)

        Output_Grad2_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer1_CH1 = process_input_lines(Output_Grad2_Layer1_CH1)

        Output_Grad3_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer1_CH0 = process_input_lines(Output_Grad3_Layer1_CH0)

        Output_Grad3_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer1_CH1 = process_input_lines(Output_Grad3_Layer1_CH1)

        Output_Grad4_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer1_CH0 = process_input_lines(Output_Grad4_Layer1_CH0)

        Output_Grad4_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer1_CH1 = process_input_lines(Output_Grad4_Layer1_CH1)

        Output_Grad5_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer1_CH0 = process_input_lines(Output_Grad5_Layer1_CH0)

        Output_Grad5_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer1_CH1 = process_input_lines(Output_Grad5_Layer1_CH1)

        Output_Grad6_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer1_CH1 = process_input_lines(Output_Grad5_Layer1_CH1)

        Output_Grad6_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer1_CH1 = process_input_lines(Output_Grad6_Layer1_CH1)

        Output_Grad7_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer1_CH0 = process_input_lines(Output_Grad7_Layer1_CH0)

        Output_Grad7_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer1_CH1 = process_input_lines(Output_Grad7_Layer1_CH1)

        Output_Grad8_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer1_CH0 = process_input_lines(Output_Grad8_Layer1_CH0)

        Output_Grad8_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer1_CH1 = process_input_lines(Output_Grad8_Layer1_CH1)

        Output_Grad1_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer1_CH0, Output_Grad1_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad2_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer1_CH0, Output_Grad2_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad3_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer1_CH0, Output_Grad3_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad4_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer1_CH0, Output_Grad4_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad5_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer1_CH0, Output_Grad5_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad6_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer1_CH0, Output_Grad6_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad7_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer1_CH0, Output_Grad7_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grad8_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer1_CH0, Output_Grad8_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Grads_Layer1 = Output_Grad1_Layer1 + Output_Grad2_Layer1 + Output_Grad3_Layer1 + Output_Grad4_Layer1 + \
                                Output_Grad5_Layer1 + Output_Grad6_Layer1 + Output_Grad7_Layer1 + Output_Grad8_Layer1    
        Output_Grad_Layer1 = torch.tensor([float(value) for value in Output_Grads_Layer1], dtype=torch.float32).reshape(8, 32, 208, 208)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer1 = (Output_Grad_Layer1).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer1_CH0_256 = process_input_lines(Weight_Gradient1_Layer1_CH0)
        print("Weight_Gradient1_Layer1_CH0 : ", len(Weight_Gradient1_Layer1_CH0))   

        Weight_Gradient2_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer1_CH0_256 = process_input_lines(Weight_Gradient2_Layer1_CH0)
        print("Weight_Gradient2_Layer1_CH0 : ", len(Weight_Gradient2_Layer1_CH0))    

        Weight_Gradient3_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer1_CH0_256 = process_input_lines(Weight_Gradient3_Layer1_CH0)
        print("Weight_Gradient3_Layer1_CH0 : ", len(Weight_Gradient3_Layer1_CH0)) 

        Weight_Gradient4_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer1_CH0_256 = process_input_lines(Weight_Gradient4_Layer1_CH0)
        print("Weight_Gradient4_Layer1_CH0 : ", len(Weight_Gradient4_Layer1_CH0)) 

        Weight_Gradient5_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer1_CH0_256 = process_input_lines(Weight_Gradient5_Layer1_CH0)
        print("Weight_Gradient5_Layer1_CH0 : ", len(Weight_Gradient5_Layer1_CH0)) 

        Weight_Gradient6_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer1_CH0_256 = process_input_lines(Weight_Gradient6_Layer1_CH0)
        print("Weight_Gradient6_Layer1_CH0 : ", len(Weight_Gradient6_Layer1_CH0)) 

        Weight_Gradient7_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer1_CH0_256 = process_input_lines(Weight_Gradient7_Layer1_CH0)
        print("Weight_Gradient7_Layer1_CH0 : ", len(Weight_Gradient7_Layer1_CH0)) 

        Weight_Gradient8_Layer1_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer1_CH0_256 = process_input_lines(Weight_Gradient8_Layer1_CH0)
        print("Weight_Gradient8_Layer1_CH0 : ", len(Weight_Gradient8_Layer1_CH0)) 

        Weight_Gradient1_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer1_CH1_256 = process_input_lines(Weight_Gradient1_Layer1_CH1)
        print("Weight_Gradient1_Layer1_CH1 : ", len(Weight_Gradient1_Layer1_CH1)) 

        Weight_Gradient2_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer1_CH1_256 = process_input_lines(Weight_Gradient2_Layer1_CH1)
        print("Weight_Gradient2_Layer1_CH1 : ", len(Weight_Gradient2_Layer1_CH1)) 

        Weight_Gradient3_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer1_CH1_256 = process_input_lines(Weight_Gradient3_Layer1_CH1)
        print("Weight_Gradient3_Layer1_CH1 : ", len(Weight_Gradient3_Layer1_CH1)) 

        Weight_Gradient4_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer1_CH1_256 = process_input_lines(Weight_Gradient4_Layer1_CH1)
        print("Weight_Gradient4_Layer1_CH1 : ", len(Weight_Gradient4_Layer1_CH1)) 

        Weight_Gradient5_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer1_CH1_256 = process_input_lines(Weight_Gradient5_Layer1_CH1)
        print("Weight_Gradient5_Layer1_CH1 : ", len(Weight_Gradient5_Layer1_CH1)) 

        Weight_Gradient6_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer1_CH1_256 = process_input_lines(Weight_Gradient6_Layer1_CH1)
        print("Weight_Gradient6_Layer1_CH1 : ", len(Weight_Gradient6_Layer1_CH1)) 

        Weight_Gradient7_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer1_CH1_256 = process_input_lines(Weight_Gradient7_Layer1_CH1)
        print("Weight_Gradient7_Layer1_CH1 : ", len(Weight_Gradient7_Layer1_CH1)) 

        Weight_Gradient8_Layer1_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer1_CH1_256 = process_input_lines(Weight_Gradient8_Layer1_CH1)
        print("Weight_Gradient8_Layer1_CH1 : ", len(Weight_Gradient8_Layer1_CH1)) 
    
        Weight_Gradient1_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer1_CH0_256, Weight_Gradient1_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient2_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer1_CH0_256, Weight_Gradient2_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient3_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer1_CH0_256, Weight_Gradient3_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient4_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer1_CH0_256, Weight_Gradient4_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient5_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer1_CH0_256, Weight_Gradient5_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient6_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer1_CH0_256, Weight_Gradient6_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient7_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer1_CH0_256, Weight_Gradient7_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient8_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer1_CH0_256, Weight_Gradient8_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient_Layer1 = [Weight_Gradient1_Layer1, Weight_Gradient2_Layer1, Weight_Gradient3_Layer1, Weight_Gradient4_Layer1, Weight_Gradient5_Layer1, 
                                Weight_Gradient6_Layer1, Weight_Gradient7_Layer1, Weight_Gradient8_Layer1]
        Weight_Gradient_Layer1 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer1)]   
        Weight_Gradient_Layer1 = torch.tensor([float(value) for value in Weight_Gradient_Layer1], dtype=torch.float32).reshape(32, 16, 3, 3)   

        #################################################
        #             Backward Layer 0 Start            #
        #################################################
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer0_CH0 = process_input_lines(Output_Grad1_Layer0_CH0)

        Output_Grad1_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad1_Layer0_CH1 = process_input_lines(Output_Grad1_Layer0_CH1)

        Output_Grad2_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer0_CH0 = process_input_lines(Output_Grad2_Layer0_CH0)

        Output_Grad2_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad2_Layer0_CH1 = process_input_lines(Output_Grad2_Layer0_CH1)

        Output_Grad3_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer0_CH0 = process_input_lines(Output_Grad3_Layer0_CH0)

        Output_Grad3_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad3_Layer0_CH1 = process_input_lines(Output_Grad3_Layer0_CH1)

        Output_Grad4_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer0_CH0 = process_input_lines(Output_Grad4_Layer0_CH0)

        Output_Grad4_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad4_Layer0_CH1 = process_input_lines(Output_Grad4_Layer0_CH1)

        Output_Grad5_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer0_CH0 = process_input_lines(Output_Grad5_Layer0_CH0)

        Output_Grad5_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer0_CH1 = process_input_lines(Output_Grad5_Layer0_CH1)

        Output_Grad6_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad5_Layer0_CH1 = process_input_lines(Output_Grad5_Layer0_CH1)

        Output_Grad6_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad6_Layer0_CH1 = process_input_lines(Output_Grad6_Layer0_CH1)

        Output_Grad7_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer0_CH0 = process_input_lines(Output_Grad7_Layer0_CH0)

        Output_Grad7_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad7_Layer0_CH1 = process_input_lines(Output_Grad7_Layer0_CH1)

        Output_Grad8_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer0_CH0 = process_input_lines(Output_Grad8_Layer0_CH0)

        Output_Grad8_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Output_Grad8_Layer0_CH1 = process_input_lines(Output_Grad8_Layer0_CH1)

        Output_Grad1_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer0_CH0, Output_Grad1_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad2_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer0_CH0, Output_Grad2_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad3_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer0_CH0, Output_Grad3_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad4_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer0_CH0, Output_Grad4_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad5_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer0_CH0, Output_Grad5_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad6_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer0_CH0, Output_Grad6_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad7_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer0_CH0, Output_Grad7_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad8_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer0_CH0, Output_Grad8_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grads_Layer0 = Output_Grad1_Layer0 + Output_Grad2_Layer0 + Output_Grad3_Layer0 + Output_Grad4_Layer0 + \
                                Output_Grad5_Layer0 + Output_Grad6_Layer0 + Output_Grad7_Layer0 + Output_Grad8_Layer0    
        Output_Grad_Layer0 = torch.tensor([float(value) for value in Output_Grads_Layer0], dtype=torch.float32).reshape(8, 16, 416, 416)

        # Gradient of Beta Calculation:
        Beta_Gradient_Layer0 = (Output_Grad_Layer0).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        Weight_Gradient1_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer0_CH0_256 = process_input_lines(Weight_Gradient1_Layer0_CH0)
        print("Weight_Gradient1_Layer0_CH0 : ", len(Weight_Gradient1_Layer0_CH0))   

        Weight_Gradient2_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer0_CH0_256 = process_input_lines(Weight_Gradient2_Layer0_CH0)
        print("Weight_Gradient2_Layer0_CH0 : ", len(Weight_Gradient2_Layer0_CH0))    

        Weight_Gradient3_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer0_CH0_256 = process_input_lines(Weight_Gradient3_Layer0_CH0)
        print("Weight_Gradient3_Layer0_CH0 : ", len(Weight_Gradient3_Layer0_CH0)) 

        Weight_Gradient4_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer0_CH0_256 = process_input_lines(Weight_Gradient4_Layer0_CH0)
        print("Weight_Gradient4_Layer0_CH0 : ", len(Weight_Gradient4_Layer0_CH0)) 

        Weight_Gradient5_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer0_CH0_256 = process_input_lines(Weight_Gradient5_Layer0_CH0)
        print("Weight_Gradient5_Layer0_CH0 : ", len(Weight_Gradient5_Layer0_CH0)) 

        Weight_Gradient6_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer0_CH0_256 = process_input_lines(Weight_Gradient6_Layer0_CH0)
        print("Weight_Gradient6_Layer0_CH0 : ", len(Weight_Gradient6_Layer0_CH0)) 

        Weight_Gradient7_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer0_CH0_256 = process_input_lines(Weight_Gradient7_Layer0_CH0)
        print("Weight_Gradient7_Layer0_CH0 : ", len(Weight_Gradient7_Layer0_CH0)) 

        Weight_Gradient8_Layer0_CH0 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer0_CH0_256 = process_input_lines(Weight_Gradient8_Layer0_CH0)
        print("Weight_Gradient8_Layer0_CH0 : ", len(Weight_Gradient8_Layer0_CH0)) 

        Weight_Gradient1_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient1_Layer0_CH1_256 = process_input_lines(Weight_Gradient1_Layer0_CH1)
        print("Weight_Gradient1_Layer0_CH1 : ", len(Weight_Gradient1_Layer0_CH1)) 

        Weight_Gradient2_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient2_Layer0_CH1_256 = process_input_lines(Weight_Gradient2_Layer0_CH1)
        print("Weight_Gradient2_Layer0_CH1 : ", len(Weight_Gradient2_Layer0_CH1)) 

        Weight_Gradient3_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient3_Layer0_CH1_256 = process_input_lines(Weight_Gradient3_Layer0_CH1)
        print("Weight_Gradient3_Layer0_CH1 : ", len(Weight_Gradient3_Layer0_CH1)) 

        Weight_Gradient4_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient4_Layer0_CH1_256 = process_input_lines(Weight_Gradient4_Layer0_CH1)
        print("Weight_Gradient4_Layer0_CH1 : ", len(Weight_Gradient4_Layer0_CH1)) 

        Weight_Gradient5_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient5_Layer0_CH1_256 = process_input_lines(Weight_Gradient5_Layer0_CH1)
        print("Weight_Gradient5_Layer0_CH1 : ", len(Weight_Gradient5_Layer0_CH1)) 

        Weight_Gradient6_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient6_Layer0_CH1_256 = process_input_lines(Weight_Gradient6_Layer0_CH1)
        print("Weight_Gradient6_Layer0_CH1 : ", len(Weight_Gradient6_Layer0_CH1)) 

        Weight_Gradient7_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient7_Layer0_CH1_256 = process_input_lines(Weight_Gradient7_Layer0_CH1)
        print("Weight_Gradient7_Layer0_CH1 : ", len(Weight_Gradient7_Layer0_CH1)) 

        Weight_Gradient8_Layer0_CH1 = Read_DDR(Rd_Address=None,  End_Offset=None)
        Weight_Gradient8_Layer0_CH1_256 = process_input_lines(Weight_Gradient8_Layer0_CH1)
        print("Weight_Gradient8_Layer0_CH1 : ", len(Weight_Gradient8_Layer0_CH1)) 
    
        Weight_Gradient1_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer0_CH0_256, Weight_Gradient1_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient2_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer0_CH0_256, Weight_Gradient2_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient3_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer0_CH0_256, Weight_Gradient3_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient4_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer0_CH0_256, Weight_Gradient4_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient5_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer0_CH0_256, Weight_Gradient5_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient6_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer0_CH0_256, Weight_Gradient6_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient7_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer0_CH0_256, Weight_Gradient7_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient8_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer0_CH0_256, Weight_Gradient8_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient_Layer0 = [Weight_Gradient1_Layer0, Weight_Gradient2_Layer0, Weight_Gradient3_Layer0, Weight_Gradient4_Layer0, Weight_Gradient5_Layer0, 
                                Weight_Gradient6_Layer0, Weight_Gradient7_Layer0, Weight_Gradient8_Layer0]
        Weight_Gradient_Layer0 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer0)]   
        Weight_Gradient_Layer0 = torch.tensor([float(value) for value in Weight_Gradient_Layer0], dtype=torch.float32).reshape(16, 3, 3, 3)   


        
    def Stop_click(self):
        print("Button 6 clicked")

    def Weight_Update_click(self):
        print("Button 7 clicked")
        
    def Reset_click(self):
        print("Button 8 clicked")
        self.L8_IRQ_canvas.itemconfig(self.L8_IRQ, fill="green")

def Read_DDR(Rd_Address, End_Address):
    d = Device("0000:08:00.0")
    bar = d.bar[2]
    Read_Data_List = []
    i=0
    for i in range(0,int((End_Address-Rd_Address)/4)): 
        Read_Data = bar.read(Rd_Address + (i*4))
        Read_Data_List.append(Read_Data)
    return Read_Data_List  

def Write_DDR(Wr_Data_List, Wr_Address):
    d = Device("0000:08:00.0")
    bar = d.bar[2]
    i = 0
    for item in Wr_Data_List:
        an_integer = int(item, 16)
        bar.write(Wr_Address +i, an_integer)
        i = i + 0x4
 
def Microcode(read_path):
    Microcode_List = []
    read = open(read_path, mode="r")
    Microcode = read.readlines()
    for value in Microcode:
        value = value.replace(',', '').replace('\n', '')
        value = int(value, 16)
        Microcode_List.append(value)
    return Microcode_List 
 
def fill_0_data(data):
    data_str = str(data).zfill(8)
    return data_str

def flip_lines(lines):
    flipped_lines = lines[::-1]
    return flipped_lines

def process_input_lines(lines):
    processed_lines = []

    for i in range(0, len(lines), 8):
        chunk = lines[i:i+8] 

        hex_lines = [fill_0_data(hex(line)[2:]) for line in chunk]

        flipped_lines = flip_lines(hex_lines)
        combined_line = ''.join(flipped_lines)

        processed_lines.append(combined_line)

    return processed_lines

def break_data_256_32(data_list, chunk_size=8):
    processed_data = []

    for data in data_list:
        output = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        processed_data.extend(output)

    return processed_data


def clean_string(text):
    if isinstance(text, str):
        return text.replace('[', '').replace(']', '').replace("'", '')
    return text

if __name__ == "__main__":
    app = App()
    app.mainloop()

