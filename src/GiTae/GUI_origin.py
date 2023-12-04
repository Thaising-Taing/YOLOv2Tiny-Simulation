import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from pypcie import Device
from ast import literal_eval
import subprocess
from  XdmaAccess import XdmaAccess
import sys
sys.path.append("/home/msis/Desktop/pcie_python/GUI")
from Pre_Processing_Scratch.Pre_Processing import *
from Pre_Processing_Scratch.Pre_Processing_Function import *
from Post_Processing_Scratch.Post_Processing_2Iterations import Post_Processing
import time
from tabulate import tabulate
import os.path 

DDR_SIZE = 0x10000


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
        print("Button 3 clicked")
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]
        #self.textbox.insert("0.0", "CTkTextbox\n\n" )

        #microcode = Microcode("mic_2iteration_forward_hex_add_0x.txt") 
        microcode = Microcode("mic_2_iteration_whole.txt")

        for i in range (0, len(microcode)):
            self.bar.write(0x4, microcode[i]) # wr mic
            self.bar.write(0x8, i) # wr addr
            self.bar.write(0x0, 0x00000012) # wr en
            self.bar.write(0x0, 0x00000010) # wr en low
        print("mic write done")    

        
    def Run_click(self):
        learning_rate = 0.001 #0.00001
        Total_epoch = 1000

        for epoch in range(Total_epoch) :
            whole_process_start = time.time()
            # start signal
            print("Button 5 clicked")

            Write_Weight(epoch)

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

            Forward(epoch)

            Backward()

            Weight_Update(learning_rate)

            whole_process_end = time.time()

            whole_process_time = whole_process_end - whole_process_start

            print("1 epoch time : ",whole_process_time)
            print("epoch : ", epoch+1, ", Loss : ", Loss)
            

        

        
    def Stop_click(self):
        print("Button 6 clicked")

    def Weight_Update_click(self):
        print("Button 7 clicked")
        
    def Reset_click(self):
        print("Button 8 clicked")
        self.L8_IRQ_canvas.itemconfig(self.L8_IRQ, fill="green")

def Read_DDR(Rd_Address, End_Address):
    device = XdmaAccess(0)
    Read_Data_List = device.read_dma(Rd_Address, ((End_Address-Rd_Address)))
    # print("Read_Data_List : ", Read_Data_List)
    device.__close__()
    return Read_Data_List  

'''
def Read_DDR(Rd_Address, End_Address):
    d = Device("0000:08:00.0")
    bar = d.bar[2]
    Read_Data_List = []
    Read_Data_List.clear()
    i=0
    for i in range(0,int((End_Address-Rd_Address)/4)): 
        Read_Data = bar.read(Rd_Address + (i*4))
        Read_Data_List.append(Read_Data)
    return Read_Data_List  
'''

def Write_DDR(Wr_Data_List, Wr_Address):
    device = XdmaAccess(0)
    data_to_write = []
    for line in Wr_Data_List:
        line_data = int(line.strip(), 16)
        data_to_write.append(line_data)
    data_to_write_array = np.array(data_to_write, dtype=np.uint32)
    device.write_dma(Wr_Address, data_to_write_array) 
    device.__close__()

'''
def Write_DDR(Wr_Data_List, Wr_Address):
    d = Device("0000:08:00.0")
    bar = d.bar[2]
    i = 0
    for item in Wr_Data_List:
        an_integer = int(item, 16)
        bar.write(Wr_Address +i, an_integer)
        i = i + 0x4
'''

def Microcode(read_path):
    Microcode_List = []
    Microcode_List.clear()
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
    processed_lines.clear()

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

'''
def backward_LightNorm(grad_output, cache):
        
        X, gamma, beta, output, output_hat, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index = cache
        B, C, H, W = X.shape        
        
        #print('grad_output', grad_output)
        # print(grad_output.shape)
        dL_dxi_hat = grad_output * gamma.view(1, -1, 1, 1)
        """
        dL_dvar = dL_dxi_hat * (X - avg) * -0.5 * torch.sqrt(scale) * torch.sqrt(scale) * torch.sqrt(scale)
        dL_dvar_tmp = torch.zeros(dL_dvar.size()).cuda()
        for idx in max_index:
            dL_dvar_tmp[idx[0], idx[1], idx[2], idx[3]] = dL_dvar[idx[0], idx[1], idx[2], idx[3]] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
        for idx in min_index:
            dL_dvar_tmp[idx[0], idx[1], idx[2], idx[3]] = dL_dvar[idx[0], idx[1], idx[2], idx[3]]
        dL_dvar = dL_dvar_tmp.sum(dim=(0, 2, 3), keepdim=True)
        """
        dL_dvar = (dL_dxi_hat * (X - avg) * -0.5 * torch.sqrt(scale) * torch.sqrt(scale) * torch.sqrt(scale)).sum(dim=(0, 2, 3), keepdim=True)
        dL_dxmax_mean = (dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
        dL_dxmin_mean = (-1 * dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
        dL_dxmax = (dL_dxmax_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
        dL_dxmin = (dL_dxmin_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
        # dL_davg = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True)
        # dL_dxi = dL_davg / (B*H*W) + dL_dxi_hat * scale
        # for idx in max_index:
        #     dL_dxi[idx[0], idx[1], idx[2], idx[3]] += grad_output[idx[0], idx[1], idx[2], idx[3]]
        # for idx in min_index:
        #     dL_dxi[idx[0], idx[1], idx[2], idx[3]] -= grad_output[idx[0], idx[1], idx[2], idx[3]] #dL_dxmax[0, idx[1], 0, 0] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
        #dL_dxi_max = dL_dxi + dL_dxmax
        #dL_dxi_min = dL_dxi + dL_dxmin
        dL_dgamma = (grad_output * output_hat).sum(dim=(0, 2, 3), keepdim=True)
        dL_dbeta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)

        dL_dgamma = dL_dgamma.view(-1)
        dL_dbeta = dL_dbeta.view(-1)
        #for idx in max_index:
        #    dL_dxi[idx[0], idx[1], idx[2], idx[3]] += dL_dxmax[0, idx[1], 0, 0] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
        #for idx in min_index:
        
        #    dL_dxi[idx[0], idx[1], idx[2], idx[3]] += dL_dxmin[0, idx[1], 0, 0] #dL_dxi_min[idx[0], idx[1], idx[2], idx[3]] #
        dL_davg = (grad_output).sum(dim=(0, 2, 3), keepdim=True)
        
        #average per channel
        avg_pc = dL_davg / (B*H*W) # write to file
        dL_dxi_ = avg_pc + grad_output


        # y = grad_output.transpose(0, 1).contiguous()  # C x B x H x W
        # y = y.view(C, num_chunks, B * H * W // num_chunks)
        # avg_max_ = y.max(-1)[0].mean(-1)  # C
        # avg_min_ = y.min(-1)[0].mean(-1)  # C
        # scale_ = 1 / ((avg_max_ - avg_min_) * scale_fix + eps)  
        # scale_ = scale_.view(1, -1, 1, 1)

        # backward coefficient
        backward_const = -1. * gamma.view(1, -1, 1, 1) / (scale + eps)   # write to file
        
        dl_dxi = dL_dxi_ * backward_const

        return dL_dgamma, dL_dbeta, avg_pc, backward_const
'''

'''
def backward_LightNorm(grad_output, cache):
        
        X, gamma, beta, output, output_hat, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index = cache
        B, C, H, W = X.shape        
        
        dL_dxi_hat = grad_output * gamma.view(1, -1, 1, 1)
        

        dL_dgamma = (grad_output * output_hat).sum(dim=(0, 2, 3), keepdim=True)
        dL_dbeta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)

        dL_dgamma = dL_dgamma.view(-1)
        dL_dbeta = dL_dbeta.view(-1)
        
        dL_davg = (grad_output).sum(dim=(0, 2, 3), keepdim=True)
        
        #average per channel
        avg_pc = dL_davg / (B*H*W) # write to file
        dL_dxi_ = avg_pc + grad_output


        backward_const = -1. * gamma.view(1, -1, 1, 1) * (scale)   # write to file
        
        dl_dxi = dL_dxi_ * backward_const

        return dL_dgamma, dL_dbeta, avg_pc, backward_const
'''


def backward_LightNorm(grad_output, cache):
    X, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index = cache
    B, C, H, W = X.shape        
    
    #print('grad_output', grad_output)
    # print(grad_output.shape)
    dL_dxi_hat = grad_output * gamma.view(1, -1, 1, 1)
    """
    dL_dvar = dL_dxi_hat * (X - avg) * -0.5 * torch.sqrt(scale) * torch.sqrt(scale) * torch.sqrt(scale)
    dL_dvar_tmp = torch.zeros(dL_dvar.size()).cuda()
    for idx in max_index:
        dL_dvar_tmp[idx[0], idx[1], idx[2], idx[3]] = dL_dvar[idx[0], idx[1], idx[2], idx[3]] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
    for idx in min_index:
        dL_dvar_tmp[idx[0], idx[1], idx[2], idx[3]] = dL_dvar[idx[0], idx[1], idx[2], idx[3]]
    dL_dvar = dL_dvar_tmp.sum(dim=(0, 2, 3), keepdim=True)
    """
    # dL_dvar = (dL_dxi_hat * (X - avg) * -0.5 * torch.sqrt(scale) * torch.sqrt(scale) * torch.sqrt(scale)).sum(dim=(0, 2, 3), keepdim=True)
    # dL_dxmax_mean = (dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
    # dL_dxmin_mean = (-1 * dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
    # dL_dxmax = (dL_dxmax_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
    # dL_dxmin = (dL_dxmin_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
    # dL_davg = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True)
    # dL_dxi = dL_davg / (B*H*W) + dL_dxi_hat * scale
    # for idx in max_index:
    #     dL_dxi[idx[0], idx[1], idx[2], idx[3]] += grad_output[idx[0], idx[1], idx[2], idx[3]]
    # for idx in min_index:
    #     dL_dxi[idx[0], idx[1], idx[2], idx[3]] -= grad_output[idx[0], idx[1], idx[2], idx[3]] #dL_dxmax[0, idx[1], 0, 0] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
    #dL_dxi_max = dL_dxi + dL_dxmax
    #dL_dxi_min = dL_dxi + dL_dxmin
    dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True)
    dL_dbeta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)

    dL_dgamma = dL_dgamma.view(-1)
    dL_dbeta = dL_dbeta.view(-1)
    
    #for idx in max_index:
    #    dL_dxi[idx[0], idx[1], idx[2], idx[3]] += dL_dxmax[0, idx[1], 0, 0] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
    #for idx in min_index:
    
    #    dL_dxi[idx[0], idx[1], idx[2], idx[3]] += dL_dxmin[0, idx[1], 0, 0] #dL_dxi_min[idx[0], idx[1], idx[2], idx[3]] #
    dL_davg = (grad_output).sum(dim=(0, 2, 3), keepdim=True)
    
    #hardware implementation
    #average per channel
    
    avg_pc = (grad_output * -1.0).sum(dim=(0, 2, 3), keepdim=True) / (B*H*W) # write to file
    dL_dxi_ = avg_pc + dL_dxi_hat
    
    # backward coefficient
    backward_const = scale * gamma.view(1, -1, 1, 1)   # write to file

     
    #final output calculation
    dL_dxi = dL_dxi_ * backward_const

    # for Software
    # for idx in max_index:
    #     dL_dxi[idx[0], idx[1], idx[2], idx[3]] += grad_output[idx[0], idx[1], idx[2], idx[3]]
    # for idx in min_index:
    #     dL_dxi[idx[0], idx[1], idx[2], idx[3]] -= grad_output[idx[0], idx[1], idx[2], idx[3]] #dL_dxmax[0, idx[1], 0, 0] #dL_dxi_max[idx[0], idx[1], idx[2], idx[3]] #
    #dL_dxi_max = dL_dxi + dL_dxmax

    return dL_dgamma, dL_dbeta, avg_pc, backward_const

def split_location(mask_location): 
    relu_mask = torch.zeros_like(mask_location)
    relu_mask[mask_location>3] = 1
    
    location = torch.zeros_like(mask_location)
    location[mask_location==0] = 0
    location[mask_location==2] = 1
    location[mask_location==1] = 2
    location[mask_location==3] = 3
    location[mask_location==4] = 0
    location[mask_location==6] = 1
    location[mask_location==5] = 2
    location[mask_location==7] = 3
    
    return relu_mask, location
    
def backward_active(gradient, relu_mask, alpha=0.1):
    dx, x = None, relu_mask
    
    dl = torch.ones_like(x)
    dl[x > 0] = alpha
    dx = gradient * dl
    
    return dx

def backward_ReLU(dout, cache, alpha=0.1):
    dx, x = None, cache
    
    dl = torch.ones_like(x)
    dl[x < 0] = alpha
    dx = dout * dl
    
    return dx

def backward_MaxPool(dout, x, layer_no=[], save_txt=False, save_hex=False, phase=[]):
    
    x = x
    dx = None

    N, C, H, W = x.shape
    stride = 2
    pool_width = 2
    pool_height = 2
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    dx = torch.zeros_like(x)
    
    backward_positions = []
    backward_positions.clear()
    
    
    for n in range(N):
        for c in range(C):
            temp_positions = []
            temp_positions.clear()
            for height in range(H_out):
                for width in range(W_out):
                    local_x = x[n, c, height * stride:height * stride + pool_height,
                            width * stride:width * stride + pool_width]
                    
                    shape_local_x = local_x.shape
                    
                    input_tensor = local_x.reshape(-1)
                    
                    # print("input_tensor",input_tensor)
                    
                    # print("input_tensor", input_tensor.shape, input_tensor)
                    local_dw = torch.zeros_like(input_tensor)

                    max_index = torch.argmax(input_tensor)
                    
                    max_value = input_tensor[max_index]
                    
                    all_max_indices = torch.nonzero(input_tensor == max_value).flatten()
                    
                    
                    last_index_of_highest_value = all_max_indices[-1].item()
                    backward_positions.append(last_index_of_highest_value)
                    temp_positions.append(last_index_of_highest_value)
                    # values, indicies = input_tensor.max(-1)
                    local_dw[last_index_of_highest_value] = dout[n, c, height, width]
                    dx[n, c, height * stride:height * stride + pool_height,
                    width * stride:width * stride + pool_width] = local_dw.reshape(shape_local_x)
    
    backward_positions = torch.tensor(backward_positions)

    return dx

def backward_MaxPool_Location(dout, Location, layer_no=[], save_txt=False, save_hex=False, phase=[]):
    
    # x, pool_param = cache

    dx = None

    N, C, H, W = Location.shape
    H = 2*H
    W = 2*W
    # stride = pool_param['stride']
    # pool_width = pool_param['pool_width']
    # pool_height = pool_param['pool_height']
    stride = 2
    pool_width = 2
    pool_height = 2
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    dx = torch.zeros(N, C, H, W)
    
    for n in range(N):
        for c in range(C):
            temp_positions = []
            for height in range(H_out):
                for width in range(W_out):
                    local_dw = torch.zeros(4)
                    local_dw[int(Location[n, c, height, width])] = dout[n, c, height, width]
                    dx[n, c, height * stride:height * stride + pool_height,
                    width * stride:width * stride + pool_width] = local_dw.reshape(2,2)
    
    return dx    

'''
def Weight_Update(lr):
    # Learning Parameters Updating
    Weight_Tensor = Convert_Tensor(Weight_Dec_List)
    Bias_Tensor = Convert_Bias_Tensor(Bias_Dec_List)
    Beta_Tensor = Convert_BN_Tensor(Beta_Dec_List)
    Gamma_Tensor = Convert_BN_Tensor(Gamma_Dec_List)
    # Temporarily Disable Gradient Computation
    with torch.no_grad():
        for i in range(8):
            Weight_Tensor[i] -= lr * Weight_Gradient[i]
            save_weights(Weight_Tensor[i], f"/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_{i}.mem")
            Beta_Tensor[i] -= lr * Beta_Gradient[i].reshape(-1) # Reshaping into 1-D Array
            save_weights(Beta_Tensor[i], f"/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_{i}.mem")
            Gamma_Tensor[i] -= lr * Gamma_Gradient[i].reshape(-1) # Reshaping into 1-D Array
            save_weights(Gamma_Tensor[i], f"/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_{i}.mem")
        
        Weight_Tensor[8] -= lr * Weight_Gradient[8]
        save_weights(Weight_Tensor[8], "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_8.mem")
        
        # Bias Update
        Bias_Tensor -= lr * Bias_Grad
        save_weights(Bias_Tensor, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Bias/Bias_8.mem")
'''


def Weight_Update(lr):
    # Weight Update
    Weight_Tensor = Convert_Tensor(Weight_List)
    Weight_Update0 = Weight_Tensor[0] - lr*Weight_Gradient_Layer0
    save_weights(Weight_Update0, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_0.mem")
    Weight_Update1 = Weight_Tensor[1] - lr*Weight_Gradient_Layer1
    save_weights(Weight_Update1, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_1.mem")
    Weight_Update2 = Weight_Tensor[2] - lr*Weight_Gradient_Layer2
    save_weights(Weight_Update2, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_2.mem")
    Weight_Update3 = Weight_Tensor[3] - lr*Weight_Gradient_Layer3
    save_weights(Weight_Update3, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_3.mem")
    Weight_Update4 = Weight_Tensor[4] - lr*Weight_Gradient_Layer4
    save_weights(Weight_Update4, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_4.mem")
    Weight_Update5 = Weight_Tensor[5] - lr*Weight_Gradient_Layer5
    save_weights(Weight_Update5, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_5.mem")
    Weight_Update6 = Weight_Tensor[6] - lr*Weight_Gradient_Layer6
    save_weights(Weight_Update6, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_6.mem")
    Weight_Update7 = Weight_Tensor[7] - lr*Weight_Gradient_Layer7
    save_weights(Weight_Update7, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_7.mem")
    Weight_Update8 = Weight_Tensor[8] - lr*Weight_Gradient_Layer8
    save_weights(Weight_Update8, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param/Conv_Weight_8.mem")
    print("Weight Update done")
    
    # Bias Update
    Bias_Dec_Tensor = torch.tensor(Bias_Dec_List, dtype=torch.float32)
    Bias_Update = Bias_Dec_Tensor - lr * Bias_Grad
    save_weights(Bias_Update, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Bias/Bias_8.mem")
    print("Bias Update done")
    
    # Beta Update: 
    Beta_Tensor = Convert_BN_Tensor(Beta_List)
    Beta_Update0 = Beta_Tensor[0] - lr*dL_dbeta_0
    save_weights(Beta_Update0, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_0.mem")
    Beta_Update1 = Beta_Tensor[1] - lr*dL_dbeta_1
    save_weights(Beta_Update1, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_1.mem")
    Beta_Update2 = Beta_Tensor[2] - lr*dL_dbeta_2
    save_weights(Beta_Update2, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_2.mem")
    Beta_Update3 = Beta_Tensor[3] - lr*dL_dbeta_3
    save_weights(Beta_Update3, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_3.mem")
    Beta_Update4 = Beta_Tensor[4] - lr*dL_dbeta_4
    save_weights(Beta_Update4, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_4.mem")
    Beta_Update5 = Beta_Tensor[5] - lr*dL_dbeta_5
    save_weights(Beta_Update5, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_5.mem")
    Beta_Update6 = Beta_Tensor[6] - lr*dL_dbeta_6
    save_weights(Beta_Update6, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_6.mem")
    Beta_Update7 = Beta_Tensor[7] - lr*dL_dbeta_7
    save_weights(Beta_Update7, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta/BN_Param_Beta_7.mem")
    print("Beta Update done")

    # Gamma Update: 
    Gamma_Tensor = Convert_BN_Tensor(Gamma_List)
    Gamma_Update0 = Gamma_Tensor[0] - lr*dL_dgamma_0
    save_weights(Gamma_Update0, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_0.mem")
    Gamma_Update1 = Gamma_Tensor[1] - lr*dL_dgamma_1
    save_weights(Gamma_Update1, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_1.mem")
    Gamma_Update2 = Gamma_Tensor[2] - lr*dL_dgamma_2
    save_weights(Gamma_Update2, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_2.mem")
    Gamma_Update3 = Gamma_Tensor[3] - lr*dL_dgamma_3
    save_weights(Gamma_Update3, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_3.mem")
    Gamma_Update4 = Gamma_Tensor[4] - lr*dL_dgamma_4
    save_weights(Gamma_Update4, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_4.mem")
    Gamma_Update5 = Gamma_Tensor[5] - lr*dL_dgamma_5
    save_weights(Gamma_Update5, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_5.mem")
    Gamma_Update6 = Gamma_Tensor[6] - lr*dL_dgamma_6
    save_weights(Gamma_Update6, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_6.mem")
    Gamma_Update7 = Gamma_Tensor[7] - lr*dL_dgamma_7
    save_weights(Gamma_Update7, "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma/BN_Param_Gamma_7.mem")
    print("Gamma Update done")

def Forward(epoch):
    global layer0_cache, layer1_cache, layer2_cache, layer3_cache, layer4_cache, layer5_cache, layer6_cache, layer7_cache, Weight_List, Weight_Dec_List, \
        Bias_Dec_List, Beta_Dec_List, Gamma_Dec_List, Loss, Beta_List, Gamma_List
    start = time.time()
    #################################################
    #                Layer 0 Start                  #
    #################################################       
    # layer0 capture interrupt
    check_irq_layer0()      
    
    global OutImage_1st_Layer0, OutImage_1st_Layer1, OutImage_1st_Layer2, OutImage_1st_Layer3, OutImage_1st_Layer4,\
    OutImage_1st_Layer5, OutImage_1st_Layer5, OutImage_1st_Layer7, OutImage_1st_Layer8, Bias_Grad
    # Layer 0
    # Read DDR & Conver Format # 512MB
    layer0_start = time.time()

    rd_start = time.time()
    print("Read DDR")
    Layer0_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x83E00000, End_Address=0x83ED0000)
    Layer0_1st_Iter_Image1_CH0_256 = process_input_lines(Layer0_1st_Iter_Image1_CH0)
    #print("ch0 image 1 : ", len(Layer0_1st_Iter_Image1_CH0)) 

    Layer0_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x83ED0000, End_Address=0x83FA0000)
    Layer0_1st_Iter_Image2_CH0_256 = process_input_lines(Layer0_1st_Iter_Image2_CH0)
    #print("ch0 image 2 : ", len(Layer0_1st_Iter_Image2_CH0))
    
    Layer0_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x83FA0000, End_Address=0x84070000)
    Layer0_1st_Iter_Image3_CH0_256 = process_input_lines(Layer0_1st_Iter_Image3_CH0)
    #print("ch0 image 3 : ", len(Layer0_1st_Iter_Image3_CH0))

    Layer0_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x84070000, End_Address=0x84140000)
    Layer0_1st_Iter_Image4_CH0_256 = process_input_lines(Layer0_1st_Iter_Image4_CH0)
    #print("ch0 image 4 : ", len(Layer0_1st_Iter_Image4_CH0))

    Layer0_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x84140000, End_Address=0x84210000)
    Layer0_1st_Iter_Image5_CH0_256 = process_input_lines(Layer0_1st_Iter_Image5_CH0)
    #print("ch0 image 5 : ", len(Layer0_1st_Iter_Image5_CH0))

    Layer0_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x84210000, End_Address=0x842E0000)
    Layer0_1st_Iter_Image6_CH0_256 = process_input_lines(Layer0_1st_Iter_Image6_CH0)
    #print("ch0 image 6 : ", len(Layer0_1st_Iter_Image6_CH0))

    Layer0_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x842E0000, End_Address=0x843B0000)
    Layer0_1st_Iter_Image7_CH0_256 = process_input_lines(Layer0_1st_Iter_Image7_CH0)
    #print("ch0 image 7 : ", len(Layer0_1st_Iter_Image7_CH0))

    Layer0_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x843B0000, End_Address=0x84480000)
    Layer0_1st_Iter_Image8_CH0_256 = process_input_lines(Layer0_1st_Iter_Image8_CH0)
    #print("ch0 image 8 : ", len(Layer0_1st_Iter_Image8_CH0))


    Layer0_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x93E00000, End_Address=0x93ED0000)
    Layer0_1st_Iter_Image1_CH1_256 = process_input_lines(Layer0_1st_Iter_Image1_CH1)
    #print("ch1 image 1 : ", len(Layer0_1st_Iter_Image1_CH1))

    Layer0_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x93ED0000, End_Address=0x93FA0000)
    Layer0_1st_Iter_Image2_CH1_256 = process_input_lines(Layer0_1st_Iter_Image2_CH1)
    #print("ch1 image 2 : ", len(Layer0_1st_Iter_Image2_CH1))

    Layer0_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x93FA0000, End_Address=0x94070000)
    Layer0_1st_Iter_Image3_CH1_256 = process_input_lines(Layer0_1st_Iter_Image3_CH1)
    #print("ch1 image 3 : ", len(Layer0_1st_Iter_Image3_CH1))

    Layer0_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x94070000, End_Address=0x94140000)
    Layer0_1st_Iter_Image4_CH1_256 = process_input_lines(Layer0_1st_Iter_Image4_CH1)
    #print("ch1 image 4 : ", len(Layer0_1st_Iter_Image4_CH1))

    Layer0_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x94140000, End_Address=0x94210000)
    Layer0_1st_Iter_Image5_CH1_256 = process_input_lines(Layer0_1st_Iter_Image5_CH1)
    #print("ch1 image 5 : ", len(Layer0_1st_Iter_Image5_CH1))

    Layer0_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x94210000, End_Address=0x942E0000)
    Layer0_1st_Iter_Image6_CH1_256 = process_input_lines(Layer0_1st_Iter_Image6_CH1)
    #print("ch1 image 6 : ", len(Layer0_1st_Iter_Image6_CH1))

    Layer0_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x942E0000, End_Address=0x943B0000)
    Layer0_1st_Iter_Image7_CH1_256 = process_input_lines(Layer0_1st_Iter_Image7_CH1)
    #print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

    Layer0_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x943B0000, End_Address=0x94480000)
    Layer0_1st_Iter_Image8_CH1_256 = process_input_lines(Layer0_1st_Iter_Image8_CH1)
    #print("ch1 image 8 : ", len(Layer0_1st_Iter_Image8_CH1))
    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer0 DDR Read Time : ", rd_time)

    
    test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer0_1st_Iter_Image1_CH0:
            test_output.write(str(item) + "\n")
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
    

    bfd_start = time.time()


    print("Convert Format")
    Output_Image1_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image1_CH0_256, Layer0_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image2_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image2_CH0_256, Layer0_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image3_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image3_CH0_256, Layer0_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image4_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image4_CH0_256, Layer0_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image5_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image5_CH0_256, Layer0_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image6_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image6_CH0_256, Layer0_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image7_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image7_CH0_256, Layer0_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Image8_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image8_CH0_256, Layer0_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)

    bfd_end = time.time()
    convert_time = bfd_end - bfd_start
    print("layer0_convert_time : ", convert_time)

    OutImages_1st_Layer0 = Output_Image1_Layer0_1st_Iter + Output_Image2_Layer0_1st_Iter + Output_Image3_Layer0_1st_Iter + Output_Image4_Layer0_1st_Iter + \
                        Output_Image5_Layer0_1st_Iter + Output_Image6_Layer0_1st_Iter + Output_Image7_Layer0_1st_Iter + Output_Image8_Layer0_1st_Iter    

    OutImage_1st_Layer0 = torch.tensor([float(value) for value in OutImages_1st_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)

    mv_start = time.time()

    # Mean, Var
    print("Calculate Mean & Var")
    Mean_1st_Layer0, Var_1st_Layer0 = Cal_mean_var.forward(OutImage_1st_Layer0)    

    Gamma_Layer0 = torch.tensor([float(value) for value in Gamma_Dec_List[0]], dtype=torch.float32).reshape(16,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer0 = Var_1st_Layer0.squeeze() * Gamma_Layer0


    Beta_List0 = torch.tensor([float(value) for value in Beta_Dec_List[0]], dtype=torch.float32).reshape(16,)

    # layer0 Caches: 
    layer0_cache = BN(OutImage_1st_Layer0, Gamma_Layer0, Beta_List0)

    Mean_1st_Layer0, Var_1st_Layer0 = Mean_Var_Dec2Bfloat(Mean_1st_Layer0, Var_1st_Layer0, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, Weight_List[0], Mean_1st_Layer0, Var_1st_Layer0, Beta_List[0], Iteration="2")
    #print("Weight_2nd_Layer0 : ", Weight_2nd_Layer0)

    mv_end = time.time()
    mv_time = mv_end - mv_start
    print("layer0_mean & var_time : ", mv_time)

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
    '''
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
    '''
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer0[0]), Wr_Address=0x80000000)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer0[1]), Wr_Address=0x90000000)

    
    resume()
    print("layer0 end")

    layer0_end = time.time()
    process = layer0_end - layer0_start
    print("layer0 process time : ", process)

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer0_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 
    
    d = Device("0000:08:00.0")
    bar = d.bar[2]
    
    data_read = open("result/layer0_result_ch0_image1.txt", mode="w+")
    i=0
    for i in range(0,int((0X4550000-0X4480000)/4) ): 
        Read_Data = bar.read(0X4480000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer0_result_ch1_image1.txt", mode="w+")
    i=0
    for i in range(0,int((0X4550000-0X4480000)/4) ): 
        Read_Data = bar.read(0X14480000 + (i*4))
        data_read.write(str(Read_Data) + "\n")     

    data_read = open("result/layer0_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X784C000-0X71CC000)/4) ): 
        Read_Data = bar.read(0X71CC000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      
    '''
        
    #################################################
    #                Layer 1 Start                  #
    #################################################
    # check Layer1 IRQ
    check_irq_otherlayer()         

    # Layer 1
    rd_start = time.time()
    layer1_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer1_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x84B00000, End_Address=0x84CA0000)
    Layer1_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer1_1st_Iter_Image1_CH0))    
    
    Layer1_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x84CA0000, End_Address=0x84E40000)
    Layer1_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer1_1st_Iter_Image2_CH0))
    
    Layer1_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x84E40000, End_Address=0x84FE0000)
    Layer1_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer1_1st_Iter_Image3_CH0))

    Layer1_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x84FE0000, End_Address=0x85180000)
    Layer1_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer1_1st_Iter_Image4_CH0))

    Layer1_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x85180000, End_Address=0x85320000)
    Layer1_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer1_1st_Iter_Image5_CH0))

    Layer1_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x85320000, End_Address=0x854C0000)
    Layer1_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer1_1st_Iter_Image6_CH0))

    Layer1_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x854C0000, End_Address=0x85660000)
    Layer1_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer1_1st_Iter_Image7_CH0))

    Layer1_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x85660000, End_Address=0x85800000)
    Layer1_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer1_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer1_1st_Iter_Image8_CH0))


    Layer1_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x94B00000, End_Address=0x94CA0000)
    Layer1_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer1_1st_Iter_Image1_CH1))

    Layer1_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x94CA0000, End_Address=0x94E40000)
    Layer1_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer1_1st_Iter_Image2_CH1))

    Layer1_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x94E40000, End_Address=0x94FE0000)
    Layer1_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer1_1st_Iter_Image3_CH1))

    Layer1_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x94FE0000, End_Address=0x95180000)
    Layer1_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer1_1st_Iter_Image4_CH1))

    Layer1_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x95180000, End_Address=0x95320000)
    Layer1_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer1_1st_Iter_Image5_CH1))

    Layer1_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x95320000, End_Address=0x954C0000)
    Layer1_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer1_1st_Iter_Image6_CH1))

    Layer1_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x954C0000, End_Address=0x95660000)
    Layer1_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer1_1st_Iter_Image7_CH1))

    Layer1_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x95660000, End_Address=0x95800000)
    Layer1_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer1_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer1_1st_Iter_Image8_CH1))
    rd_end = time.time()
    rd_time = rd_end - rd_start
    #print("Layer1 DDR Read Time : ", rd_time)

    
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
    

    bfd_start = time.time()

    print("Convert Format")
    Output_Image1_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image1_CH0_256, Layer1_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image2_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image2_CH0_256, Layer1_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image3_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image3_CH0_256, Layer1_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image4_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image4_CH0_256, Layer1_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image5_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image5_CH0_256, Layer1_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image6_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image6_CH0_256, Layer1_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image7_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image7_CH0_256, Layer1_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
    Output_Image8_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image8_CH0_256, Layer1_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)

    bfd_end = time.time()
    convert_time = bfd_end - bfd_start
    print("layer1_convert_time : ", convert_time)
    
    OutImages_1st_Layer1 = Output_Image1_Layer1_1st_Iter + Output_Image2_Layer1_1st_Iter + Output_Image3_Layer1_1st_Iter + Output_Image4_Layer1_1st_Iter + \
                        Output_Image5_Layer1_1st_Iter + Output_Image6_Layer1_1st_Iter + Output_Image7_Layer1_1st_Iter + Output_Image8_Layer1_1st_Iter    

    OutImage_1st_Layer1 = torch.tensor([float(value) for value in OutImages_1st_Layer1], dtype=torch.float32).reshape(8, 32, 208, 208)

    mv_start = time.time()

    # Mean, Var
    print("Calculate Mean & Var")
    Mean_1st_Layer1, Var_1st_Layer1 = Cal_mean_var.forward(OutImage_1st_Layer1)

    # gamma_dec_list1 = []
    # beta_dec_list1 = []
    # gamma_dec_list1.clear()
    # beta_dec_list1.clear()
    
    # for Value in Gamma_List[1][:(len(Gamma_List[1]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list1.append(str(Decimal))   

    # for Value in Beta_List[1][:(len(Beta_List[1]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list1.append(str(Decimal))          



    Gamma_Layer1 = torch.tensor([float(value) for value in Gamma_Dec_List[1]], dtype=torch.float32).reshape(32,)
    

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer1 = Var_1st_Layer1.squeeze() * Gamma_Layer1



    Beta_List1 = torch.tensor([float(value) for value in Beta_Dec_List[1]], dtype=torch.float32).reshape(32,)

    layer1_cache = BN(OutImage_1st_Layer1, Gamma_Layer1, Beta_List1)

    Mean_1st_Layer1, Var_1st_Layer1 = Mean_Var_Dec2Bfloat(Mean_1st_Layer1, Var_1st_Layer1, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, Weight_List[1], Mean_1st_Layer1, Var_1st_Layer1, Beta_List[1], Iteration="2")
    
    mv_end = time.time()
    mv_time = mv_end - mv_start
    print("layer1_mean & var_time : ", mv_time)

    # data_read_mean_var = "result/layer1_mean_var.txt"
    # with open(data_read_mean_var, mode="w") as output_file:
    #     for sublist in Weight_2nd_Layer1:
    #         cleaned_sublist = [clean_string(item) for item in sublist]
    #         output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")      
    # output_file.close()          

    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer1[0]), Wr_Address=0x80000A00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer1[1]), Wr_Address=0x90000A00)

    layer1_end = time.time()
    layer1_process = layer1_end - layer1_start
    print("Layer1 process time : ", layer1_process)

    d = Device("0000:08:00.0")
    bar = d.bar[0]
    
    resume()
    print("layer1 end")

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer1_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer1_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X5B40000-0X5800000)/4) ): 
        Read_Data = bar.read(0X5800000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer1_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X5B40000-0X5800000)/4) ): 
        Read_Data = bar.read(0X15800000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    data_read = open("result/layer1_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X7B8C000-0X784C000)/4) ): 
        Read_Data = bar.read(0X784C000 + (i*4))
        data_read.write(str(Read_Data) + "\n")     
    '''   

    #################################################
    #                Layer 2 Start                  #
    #################################################
    # check Layer2 IRQ
    check_irq_otherlayer()
    
    # Layer 2
    layer2_start = time.time()
    rd_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer2_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x85B40000, End_Address=0x85C10000)
    Layer2_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer2_1st_Iter_Image1_CH0))     

    Layer2_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x85C10000, End_Address=0x85CE0000)
    Layer2_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer2_1st_Iter_Image2_CH0))
    
    Layer2_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x85CE0000, End_Address=0x85DB0000)
    Layer2_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer2_1st_Iter_Image3_CH0))

    Layer2_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x85DB0000, End_Address=0x85E80000)
    Layer2_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer2_1st_Iter_Image4_CH0))

    Layer2_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x85E80000, End_Address=0x85F50000)
    Layer2_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer2_1st_Iter_Image5_CH0))

    Layer2_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x85F50000, End_Address=0x86020000)
    Layer2_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer2_1st_Iter_Image6_CH0))

    Layer2_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86020000, End_Address=0x860F0000)
    Layer2_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer2_1st_Iter_Image7_CH0))

    Layer2_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x860F0000, End_Address=0x861C0000)
    Layer2_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer2_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer2_1st_Iter_Image8_CH0))


    Layer2_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x95B40000, End_Address=0x95C10000)
    Layer2_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer2_1st_Iter_Image1_CH1))

    Layer2_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x95C10000, End_Address=0x95CE0000)
    Layer2_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer2_1st_Iter_Image2_CH1))

    Layer2_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x95CE0000, End_Address=0x95DB0000)
    Layer2_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer2_1st_Iter_Image3_CH1))

    Layer2_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x95DB0000, End_Address=0x95E80000)
    Layer2_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer2_1st_Iter_Image4_CH1))

    Layer2_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x95E80000, End_Address=0x95F50000)
    Layer2_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer2_1st_Iter_Image5_CH1))

    Layer2_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x95F50000, End_Address=0x96020000)
    Layer2_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer2_1st_Iter_Image6_CH1))

    Layer2_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96020000, End_Address=0x960F0000)
    Layer2_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer2_1st_Iter_Image7_CH1))

    Layer2_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x960F0000, End_Address=0x961C0000)
    Layer2_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer2_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer2_1st_Iter_Image8_CH1))
    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer2 Read DDR Time : ", rd_time)

    
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
    

    bfd_start = time.time()

    print("Convert Format")
    Output_Image1_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image1_CH0_256, Layer2_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image2_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image2_CH0_256, Layer2_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image3_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image3_CH0_256, Layer2_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image4_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image4_CH0_256, Layer2_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image5_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image5_CH0_256, Layer2_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image6_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image6_CH0_256, Layer2_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image7_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image7_CH0_256, Layer2_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
    Output_Image8_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image8_CH0_256, Layer2_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)

    bfd_end = time.time()
    convert_time = bfd_end - bfd_start
    print("layer2_convert_time : ", convert_time)

    OutImages_1st_Layer2 = Output_Image1_Layer2_1st_Iter + Output_Image2_Layer2_1st_Iter + Output_Image3_Layer2_1st_Iter + Output_Image4_Layer2_1st_Iter + \
                        Output_Image5_Layer2_1st_Iter + Output_Image6_Layer2_1st_Iter + Output_Image7_Layer2_1st_Iter + Output_Image8_Layer2_1st_Iter    

    OutImage_1st_Layer2 = torch.tensor([float(value) for value in OutImages_1st_Layer2], dtype=torch.float32).reshape(8, 64, 104, 104)

    mv_start = time.time()
    # Mean, Var
    print("Calculate Mean & Var")
    Mean_1st_Layer2, Var_1st_Layer2 = Cal_mean_var.forward(OutImage_1st_Layer2)

    # gamma_dec_list2 = []
    # beta_dec_list2 = []
    # gamma_dec_list2.clear()
    # beta_dec_list2.clear()
    
    # for Value in Gamma_List[2][:(len(Gamma_List[2]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list2.append(str(Decimal))   

    # for Value in Beta_List[2][:(len(Beta_List[2]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list2.append(str(Decimal))  


    Gamma_Layer2 = torch.tensor([float(value) for value in Gamma_Dec_List[2]], dtype=torch.float32).reshape(64,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer2 = Var_1st_Layer2.squeeze() * Gamma_Layer2



    Beta_List2 = torch.tensor([float(value) for value in Beta_Dec_List[2]], dtype=torch.float32).reshape(64,)

    layer2_cache = BN(OutImage_1st_Layer2, Gamma_Layer2, Beta_List2)

    Mean_1st_Layer2, Var_1st_Layer2 = Mean_Var_Dec2Bfloat(Mean_1st_Layer2, Var_1st_Layer2, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, Weight_List[2], Mean_1st_Layer2, Var_1st_Layer2, Beta_List[2], Iteration="2")

    mv_end = time.time()
    mv_time = mv_end - mv_start
    print("layer1_mean & var_time : ", mv_time)
    
    data_read_mean_var = "result/layer2_mean_var.txt"
    with open(data_read_mean_var, mode="w") as output_file:
        for sublist in Weight_2nd_Layer2:
            cleaned_sublist = [clean_string(item) for item in sublist]
            output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")     
    output_file.close()           

    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer2[0]), Wr_Address=0x80001E00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer2[1]), Wr_Address=0x90001E00)
    
    layer2_end = time.time()
    layer2_process = layer2_end - layer2_start
    print("Layer2 process time : ", layer2_process)

    d = Device("0000:08:00.0")
    bar = d.bar[0]
    
    resume()
    print("layer2 end")

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer_2_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer2_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X6360000-0X61C0000)/4) ): 
        Read_Data = bar.read(0X61C0000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer2_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X6360000-0X61C0000)/4) ): 
        Read_Data = bar.read(0X161C0000 + (i*4))
        data_read.write(str(Read_Data) + "\n")

    data_read = open("result/layer2_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X7D2C000-0X7B8C000)/4) ): 
        Read_Data = bar.read(0X7B8C000 + (i*4))
        data_read.write(str(Read_Data) + "\n")     
    '''


    #################################################
    #                Layer 3 Start                  #
    #################################################
    # check Layer3 IRQ
    check_irq_otherlayer()

    # Layer 3
    layer3_start = time.time()
    rd_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer3_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86360000, End_Address=0x863C8000)
    Layer3_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer3_1st_Iter_Image1_CH0))     

    Layer3_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x863C8000, End_Address=0x86430000)
    Layer3_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer3_1st_Iter_Image2_CH0))
    
    Layer3_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86430000, End_Address=0x86498000)
    Layer3_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer3_1st_Iter_Image3_CH0))

    Layer3_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86498000, End_Address=0x86500000)
    Layer3_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer3_1st_Iter_Image4_CH0))

    Layer3_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86500000, End_Address=0x86568000)
    Layer3_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer3_1st_Iter_Image5_CH0))

    Layer3_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86568000, End_Address=0x865D0000)
    Layer3_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer3_1st_Iter_Image6_CH0))

    Layer3_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x865D0000, End_Address=0x86638000)
    Layer3_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer3_1st_Iter_Image7_CH0))

    Layer3_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86638000, End_Address=0x866A0000)
    Layer3_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer3_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer3_1st_Iter_Image8_CH0))


    Layer3_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96360000, End_Address=0x963C8000)
    Layer3_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer3_1st_Iter_Image1_CH1))

    Layer3_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x963C8000, End_Address=0x96430000)
    Layer3_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer3_1st_Iter_Image2_CH1))

    Layer3_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96430000, End_Address=0x96498000)
    Layer3_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer3_1st_Iter_Image3_CH1))

    Layer3_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96498000, End_Address=0x96500000)
    Layer3_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer3_1st_Iter_Image4_CH1))

    Layer3_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96500000, End_Address=0x96568000)
    Layer3_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer3_1st_Iter_Image5_CH1))

    Layer3_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96568000, End_Address=0x965D0000)
    Layer3_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer3_1st_Iter_Image6_CH1))

    Layer3_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x965D0000, End_Address=0x96638000)
    Layer3_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer3_1st_Iter_Image7_CH1))

    Layer3_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96638000, End_Address=0x966A0000)
    Layer3_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer3_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer3_1st_Iter_Image8_CH1))

    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer3 DDR Read Time : ", rd_time)

    
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
    Mean_1st_Layer3, Var_1st_Layer3 = Cal_mean_var.forward(OutImage_1st_Layer3)

    # gamma_dec_list3 = []
    # beta_dec_list3 = []
    # gamma_dec_list3.clear()
    # beta_dec_list3.clear()
    
    # for Value in Gamma_List[3][:(len(Gamma_List[3]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list3.append(str(Decimal))   

    # for Value in Beta_List[3][:(len(Beta_List[3]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list3.append(str(Decimal))  


    Gamma_Layer3 = torch.tensor([float(value) for value in Gamma_Dec_List[3]], dtype=torch.float32).reshape(128,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer3 = Var_1st_Layer3.squeeze() * Gamma_Layer3



    Beta_List3 = torch.tensor([float(value) for value in Beta_Dec_List[3]], dtype=torch.float32).reshape(128,)

    layer3_cache = BN(OutImage_1st_Layer3, Gamma_Layer3, Beta_List3)

    Mean_1st_Layer3, Var_1st_Layer3 = Mean_Var_Dec2Bfloat(Mean_1st_Layer3, Var_1st_Layer3, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, Weight_List[3], Mean_1st_Layer3, Var_1st_Layer3, Beta_List[3], Iteration="2")
    
    data_read_mean_var = "result/layer3_mean_var.txt"
    with open(data_read_mean_var, mode="w") as output_file:
        for sublist in Weight_2nd_Layer3:
            cleaned_sublist = [clean_string(item) for item in sublist]
            output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        

    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer3[0]), Wr_Address=0x80006E00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer3[1]), Wr_Address=0x90006E00)

    layer3_end = time.time()
    layer3_process = layer3_end - layer3_start
    print("Layer3 process time : ", layer3_process)
    
    
    d = Device("0000:08:00.0")
    bar = d.bar[0]
    
    resume()
    #print(irq_val)

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer3_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer3_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X6770000-0X66A0000)/4) ): 
        Read_Data = bar.read(0X66A0000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer3_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X6770000-0X66A0000)/4) ): 
        Read_Data = bar.read(0X166A0000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    data_read = open("result/layer3_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X7DFC000-0X7D2C000)/4) ): 
        Read_Data = bar.read(0X7D2C000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      
    '''

    #################################################
    #                Layer 4 Start                  #
    #################################################
    # check Layer4 IRQ
    check_irq_otherlayer()

    # Layer 4
    Layer4_start = time.time()
    rd_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer4_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86770000, End_Address=0x867A4000)
    Layer4_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer4_1st_Iter_Image1_CH0))     

    Layer4_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x867A4000, End_Address=0x867D8000)
    Layer4_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer4_1st_Iter_Image2_CH0))
    
    Layer4_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x867D8000, End_Address=0x8680C000)
    Layer4_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer4_1st_Iter_Image3_CH0))

    Layer4_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x8680C000, End_Address=0x86840000)
    Layer4_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer4_1st_Iter_Image4_CH0))

    Layer4_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86840000, End_Address=0x86874000)
    Layer4_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer4_1st_Iter_Image5_CH0))

    Layer4_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86874000, End_Address=0x868A8000)
    Layer4_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer4_1st_Iter_Image6_CH0))

    Layer4_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x868A8000, End_Address=0x868DC000)
    Layer4_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer4_1st_Iter_Image7_CH0))

    Layer4_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x868DC000, End_Address=0x86910000)
    Layer4_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer4_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer4_1st_Iter_Image8_CH0))


    Layer4_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96770000, End_Address=0x967A4000)
    Layer4_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer4_1st_Iter_Image1_CH1))

    Layer4_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x967A4000, End_Address=0x967D8000)
    Layer4_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer4_1st_Iter_Image2_CH1))

    Layer4_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x967D8000, End_Address=0x9680C000)
    Layer4_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer4_1st_Iter_Image3_CH1))

    Layer4_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x9680C000, End_Address=0x96840000)
    Layer4_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer4_1st_Iter_Image4_CH1))

    Layer4_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96840000, End_Address=0x96874000)
    Layer4_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer4_1st_Iter_Image5_CH1))

    Layer4_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96874000, End_Address=0x968A8000)
    Layer4_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer4_1st_Iter_Image6_CH1))

    Layer4_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x968A8000, End_Address=0x968DC000)
    Layer4_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer4_1st_Iter_Image7_CH1))

    Layer4_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x968DC000, End_Address=0x96910000)
    Layer4_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer4_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer4_1st_Iter_Image8_CH1))

    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer4 Read DDR Time : ", rd_time)

    '''
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
    '''
    

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
    Mean_1st_Layer4, Var_1st_Layer4 = Cal_mean_var.forward(OutImage_1st_Layer4)

    # gamma_dec_list4 = []
    # beta_dec_list4 = []
    # gamma_dec_list4.clear()
    # beta_dec_list4.clear()
    
    # for Value in Gamma_List[4][:(len(Gamma_List[4]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list4.append(str(Decimal))   

    # for Value in Beta_List[4][:(len(Beta_List[4]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list4.append(str(Decimal))  


    Gamma_Layer4 = torch.tensor([float(value) for value in Gamma_Dec_List[4]], dtype=torch.float32).reshape(256,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer4 = Var_1st_Layer4.squeeze() * Gamma_Layer4


    Beta_List4 = torch.tensor([float(value) for value in Beta_Dec_List[4]], dtype=torch.float32).reshape(256,)

    layer4_cache = BN(OutImage_1st_Layer4, Gamma_Layer4, Beta_List4)


    Mean_1st_Layer4, Var_1st_Layer4 = Mean_Var_Dec2Bfloat(Mean_1st_Layer4, Var_1st_Layer4, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, Weight_List[4], Mean_1st_Layer4, Var_1st_Layer4, Beta_List[4], Iteration="2")
    
    data_read_mean_var = "result/layer4_mean_var.txt"
    with open(data_read_mean_var, mode="w") as output_file:
        for sublist in Weight_2nd_Layer4:
            cleaned_sublist = [clean_string(item) for item in sublist]
            output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        

    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer4[0]), Wr_Address=0x8001AE00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer4[1]), Wr_Address=0x9001AE00)

    layer4_end = time.time()
    layer4_process = layer4_end - Layer4_start
    print("Layer4 process time : ", layer4_process)
    
    d = Device("0000:08:00.0")
    bar = d.bar[0]
    
    resume()
    #print(irq_val)


    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer4_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer4_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X6978000-0X6910000)/4) ): 
        Read_Data = bar.read(0X6910000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer4_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X6978000-0X6910000)/4) ): 
        Read_Data = bar.read(0X16910000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    data_read = open("result/layer4_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X7E64000-0X7DFC000)/4) ): 
        Read_Data = bar.read(0X7DFC000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      
    '''

    #################################################
    #                Layer 5 Start                  #
    #################################################
    # check Layer5 IRQ
    check_irq_otherlayer()

    # Layer 5
    Layer5_start = time.time()
    rd_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer5_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86978000, End_Address=0x86992000)
    Layer5_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer5_1st_Iter_Image1_CH0))     

    Layer5_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86992000, End_Address=0x869AC000)
    Layer5_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer5_1st_Iter_Image2_CH0))
    
    Layer5_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x869AC000, End_Address=0x869C6000)
    Layer5_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer5_1st_Iter_Image3_CH0))

    Layer5_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x869C6000, End_Address=0x869E0000)
    Layer5_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer5_1st_Iter_Image4_CH0))

    Layer5_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x869E0000, End_Address=0x869FA000)
    Layer5_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer5_1st_Iter_Image5_CH0))

    Layer5_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x869FA000, End_Address=0x86A14000)
    Layer5_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer5_1st_Iter_Image6_CH0))

    Layer5_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86A14000, End_Address=0x86A2E000)
    Layer5_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer5_1st_Iter_Image7_CH0))

    Layer5_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86A2E000, End_Address=0x86A48000)
    Layer5_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer5_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer5_1st_Iter_Image8_CH0))


    Layer5_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96978000, End_Address=0x96992000)
    Layer5_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer5_1st_Iter_Image1_CH1))

    Layer5_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96992000, End_Address=0x969AC000)
    Layer5_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer5_1st_Iter_Image2_CH1))

    Layer5_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x969AC000, End_Address=0x969C6000)
    Layer5_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer5_1st_Iter_Image3_CH1))

    Layer5_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x969C6000, End_Address=0x969E0000)
    Layer5_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer5_1st_Iter_Image4_CH1))

    Layer5_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x969E0000, End_Address=0x969FA000)
    Layer5_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer5_1st_Iter_Image5_CH1))

    Layer5_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x969FA000, End_Address=0x96A14000)
    Layer5_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer5_1st_Iter_Image6_CH1))

    Layer5_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96A14000, End_Address=0x96A2E000)
    Layer5_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer5_1st_Iter_Image7_CH1))

    Layer5_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96A2E000, End_Address=0x96A48000)
    Layer5_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer5_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer5_1st_Iter_Image8_CH1))

    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer5 Read DDR Time : ", rd_time)

    '''
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
    '''
    

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
    Mean_1st_Layer5, Var_1st_Layer5 = Cal_mean_var.forward(OutImage_1st_Layer5)

    # gamma_dec_list5 = []
    # beta_dec_list5 = []
    # gamma_dec_list5.clear()
    # beta_dec_list5.clear()
    
    # for Value in Gamma_List[5][:(len(Gamma_List[5]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list5.append(str(Decimal))   

    # for Value in Beta_List[5][:(len(Beta_List[5]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list5.append(str(Decimal))  

    Gamma_Layer5 = torch.tensor([float(value) for value in Gamma_Dec_List[5]], dtype=torch.float32).reshape(512,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer5 = Var_1st_Layer5.squeeze() * Gamma_Layer5



    Beta_List5 = torch.tensor([float(value) for value in Beta_Dec_List[5]], dtype=torch.float32).reshape(512,)

    layer5_cache = BN(OutImage_1st_Layer5, Gamma_Layer5, Beta_List5)

    Mean_1st_Layer5, Var_1st_Layer5 = Mean_Var_Dec2Bfloat(Mean_1st_Layer5, Var_1st_Layer5, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, Weight_List[5], Mean_1st_Layer5, Var_1st_Layer5, Beta_List[5], Iteration="2")
    
    data_read_mean_var = "result/layer5_mean_var.txt"
    with open(data_read_mean_var, mode="w") as output_file:
        for sublist in Weight_2nd_Layer5:
            cleaned_sublist = [clean_string(item) for item in sublist]
            output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        

    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer5[0]), Wr_Address=0x8006AE00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer5[1]), Wr_Address=0x9006AE00)

    layer5_end = time.time()
    layer5_process = layer5_end - Layer5_start
    print("Layer5 process time : ", layer5_process)
    
    d = Device("0000:08:00.0")
    bar = d.bar[0]
    
    resume()
    #print(irq_val)

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer5_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer5_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
        Read_Data = bar.read(0X6A48000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer5_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
        Read_Data = bar.read(0X16A48000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    data_read = open("result/layer5_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X7F34000-0X7E64000)/4) ): 
        Read_Data = bar.read(0X7E64000 + (i*4))
        data_read.write(str(Read_Data) + "\n")     
    '''

    #################################################
    #                Layer 6 Start                  #
    #################################################
    # check Layer6 IRQ
    check_irq_otherlayer()

    # Layer 6
    Layer6_start = time.time()
    rd_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer6_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86B18000, End_Address=0x86B4C000)
    Layer6_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer6_1st_Iter_Image1_CH0))     

    Layer6_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86B4C000, End_Address=0x86B80000)
    Layer6_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer6_1st_Iter_Image2_CH0))
    
    Layer6_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86B80000, End_Address=0x86BB4000)
    Layer6_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer6_1st_Iter_Image3_CH0))

    Layer6_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86BB4000, End_Address=0x86BE8000)
    Layer6_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer6_1st_Iter_Image4_CH0))

    Layer6_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86BE8000, End_Address=0x86C1C000)
    Layer6_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer6_1st_Iter_Image5_CH0))

    Layer6_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86C1C000, End_Address=0x86C50000)
    Layer6_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer6_1st_Iter_Image6_CH0))

    Layer6_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86C50000, End_Address=0x86C84000)
    Layer6_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer6_1st_Iter_Image7_CH0))

    Layer6_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86C84000, End_Address=0x86CB8000)
    Layer6_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer6_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer6_1st_Iter_Image8_CH0))


    Layer6_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96B18000, End_Address=0x96B4C000)
    Layer6_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer6_1st_Iter_Image1_CH1))

    Layer6_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96B4C000, End_Address=0x96B80000)
    Layer6_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer6_1st_Iter_Image2_CH1))

    Layer6_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96B80000, End_Address=0x96BB4000)
    Layer6_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer6_1st_Iter_Image3_CH1))

    Layer6_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96BB4000, End_Address=0x96BE8000)
    Layer6_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer6_1st_Iter_Image4_CH1))

    Layer6_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96BE8000, End_Address=0x96C1C000)
    Layer6_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer6_1st_Iter_Image5_CH1))

    Layer6_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96C1C000, End_Address=0x96C50000)
    Layer6_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer6_1st_Iter_Image6_CH1))

    Layer6_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96C50000, End_Address=0x96C84000)
    Layer6_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer6_1st_Iter_Image7_CH1))

    Layer6_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96C84000, End_Address=0x96CB8000)
    Layer6_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer6_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer6_1st_Iter_Image8_CH1))

    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer6 DDR Read Time : ", rd_time)

    '''
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
    '''
    

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
    Mean_1st_Layer6, Var_1st_Layer6 = Cal_mean_var.forward(OutImage_1st_Layer6)

    # gamma_dec_list6 = []
    # beta_dec_list6 = []
    # gamma_dec_list6.clear()
    # beta_dec_list6.clear()
    
    # for Value in Gamma_List[6][:(len(Gamma_List[6]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list6.append(str(Decimal))   

    # for Value in Beta_List[6][:(len(Beta_List[6]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list6.append(str(Decimal))  


    Gamma_Layer6 = torch.tensor([float(value) for value in Gamma_Dec_List[6]], dtype=torch.float32).reshape(1024,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer6 = Var_1st_Layer6.squeeze() * Gamma_Layer6



    Beta_List6 = torch.tensor([float(value) for value in Beta_Dec_List[6]], dtype=torch.float32).reshape(1024,)

    layer6_cache = BN(OutImage_1st_Layer6, Gamma_Layer6, Beta_List6)

    Mean_1st_Layer6, Var_1st_Layer6 = Mean_Var_Dec2Bfloat(Mean_1st_Layer6, Var_1st_Layer6, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, Weight_List[6], Mean_1st_Layer6, Var_1st_Layer6, Beta_List[6], Iteration="2")
    
    data_read_mean_var = "result/layer6_mean_var.txt"
    with open(data_read_mean_var, mode="w") as output_file:
        for sublist in Weight_2nd_Layer6:
            cleaned_sublist = [clean_string(item) for item in sublist]
            output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        

    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer6[0]), Wr_Address=0x801AAE00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer6[1]), Wr_Address=0x901AAE00)

    layer6_end = time.time()
    layer6_process = layer6_end - Layer6_start
    print("Layer6 process time : ", layer6_process)
    
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    #print(irq_val)

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer6_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer6_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
        Read_Data = bar.read(0X6CB8000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer6_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
        Read_Data = bar.read(0X16CB8000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    data_read = open("result/layer6_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X80D4000-0X7F34000)/4) ): 
        Read_Data = bar.read(0X7F34000 + (i*4))
        data_read.write(str(Read_Data) + "\n")       
    '''

    #################################################
    #                Layer 7 Start                  #
    #################################################
    # check Layer7 IRQ
    check_irq_otherlayer()

    # Layer 7
    Layer7_start = time.time()
    rd_start = time.time()
    # Read DDR & Conver Format # 512MB
    print("Read DDR")
    Layer7_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86E58000, End_Address=0x86E8C000)
    Layer7_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer7_1st_Iter_Image1_CH0))     

    Layer7_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86E8C000, End_Address=0x86EC0000)
    Layer7_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer7_1st_Iter_Image2_CH0))
    
    Layer7_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86EC0000, End_Address=0x86EF4000)
    Layer7_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer7_1st_Iter_Image3_CH0))

    Layer7_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86EF4000, End_Address=0x86F28000)
    Layer7_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer7_1st_Iter_Image4_CH0))

    Layer7_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86F28000, End_Address=0x86F5C000)
    Layer7_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer7_1st_Iter_Image5_CH0))

    Layer7_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86F5C000, End_Address=0x86F90000)
    Layer7_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer7_1st_Iter_Image6_CH0))

    Layer7_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86F90000, End_Address=0x86FC4000)
    Layer7_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer7_1st_Iter_Image7_CH0))

    Layer7_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86FC4000, End_Address=0x86FF8000)
    Layer7_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer7_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer7_1st_Iter_Image8_CH0))


    Layer7_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96E58000, End_Address=0x96E8C000)
    Layer7_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image1_CH1))
    #print("ch1 image 1 : ", len(Layer7_1st_Iter_Image1_CH1))

    Layer7_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96E8C000, End_Address=0x96EC0000)
    Layer7_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer7_1st_Iter_Image2_CH1))

    Layer7_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96EC0000, End_Address=0x96EF4000)
    Layer7_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer7_1st_Iter_Image3_CH1))

    Layer7_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96EF4000, End_Address=0x96F28000)
    Layer7_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer7_1st_Iter_Image4_CH1))

    Layer7_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96F28000, End_Address=0x96F5C000)
    Layer7_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer7_1st_Iter_Image5_CH1))

    Layer7_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96F5C000, End_Address=0x96F90000)
    Layer7_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer7_1st_Iter_Image6_CH1))

    Layer7_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96F90000, End_Address=0x96FC4000)
    Layer7_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer7_1st_Iter_Image7_CH1))

    Layer7_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96FC4000, End_Address=0x96FF8000)
    Layer7_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer7_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer7_1st_Iter_Image8_CH1))

    rd_end = time.time()
    rd_time = rd_end - rd_start
    print("Layer7 Read DDR Time : ", rd_time)

    '''
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
    '''

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
    Mean_1st_Layer7, Var_1st_Layer7 = Cal_mean_var.forward(OutImage_1st_Layer7)

    # gamma_dec_list7 = []
    # beta_dec_list7 = []
    # gamma_dec_list7.clear()
    # beta_dec_list7.clear()
    
    # for Value in Gamma_List[7][:(len(Gamma_List[7]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     gamma_dec_list7.append(str(Decimal))   

    # for Value in Beta_List[7][:(len(Beta_List[7]))]:
    #     Hexadecimal = str(Value) + "0000"
    #     Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
    #     Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
    #     beta_dec_list7.append(str(Decimal))  


    Gamma_Layer7 = torch.tensor([float(value) for value in Gamma_Dec_List[7]], dtype=torch.float32).reshape(1024,)

    # Squeeze to remove the dimension but keeping the same data ordering
    Var_1st_Layer7 = Var_1st_Layer7.squeeze() * Gamma_Layer7



    Beta_List7 = torch.tensor([float(value) for value in Beta_Dec_List[7]], dtype=torch.float32).reshape(1024,)

    layer7_cache = BN(OutImage_1st_Layer7, Gamma_Layer7, Beta_List7)

    Mean_1st_Layer7, Var_1st_Layer7 = Mean_Var_Dec2Bfloat(Mean_1st_Layer7, Var_1st_Layer7, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], Mean_1st_Layer7, Var_1st_Layer7, Beta_List[7], Iteration="2")
    '''
    data_read_mean_var = "result/layer7_mean_var.txt"
    with open(data_read_mean_var, mode="w") as output_file:
        for sublist in Weight_2nd_Layer7:
            cleaned_sublist = [clean_string(item) for item in sublist]
            output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
    '''
    # Write DDR
    print("Write DDR")
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer7[0]), Wr_Address=0x806AAE00)
    Write_DDR(Break_FlipHex_256To32(Weight_2nd_Layer7[1]), Wr_Address=0x906AAE00)

    layer7_end = time.time()
    layer7_process = layer7_end - Layer7_start
    print("Layer7 process time : ", layer7_process)
    
    d = Device("0000:08:00.0")
    bar = d.bar[0]
    
    resume()
    #print(irq_val)

    '''
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/layer7_slave.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    d = Device("0000:08:00.0")
    bar = d.bar[2]

    data_read = open("result/layer7_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
        Read_Data = bar.read(0X6FF8000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer7_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
        Read_Data = bar.read(0X16FF8000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

    data_read = open("result/layer7_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X8274000-0X80D4000)/4) ): 
        Read_Data = bar.read(0X80D4000 + (i*4))
        data_read.write(str(Read_Data) + "\n")     
    '''    
    
    end = time.time()
    process_time = (end-start)/60
    print(f'Whole Process: {process_time} mn')
    #################################################
    #                Layer 8 Start                  #
    #################################################
    # check Layer8 IRQ
    check_irq_otherlayer()

    # Post-Processing Pre-Defined Conditions
    #Post_Start_Signal = "1"

    # OutputImage from Hardware
    print("Read DDR")

    # Post Processing
    #if Post_Start_Signal == "1" or Post_Start_Signal == "1".zfill(4) or Post_Start_Signal == "1".zfill(16):  

    # Layer 8
    
    # Read DDR & Conver Format # 512MB
    
    Layer8_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x87198000, End_Address=0x8719E800)
    Layer8_1st_Iter_Image1_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image1_CH0))   
    #print("ch0 image 1 : ", len(Layer8_1st_Iter_Image1_CH0))     

    Layer8_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x8719E800, End_Address=0x871A5000)
    Layer8_1st_Iter_Image2_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image2_CH0))
    #print("ch0 image 2 : ", len(Layer8_1st_Iter_Image2_CH0))

    Layer8_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x871A5000, End_Address=0x871AB800)
    Layer8_1st_Iter_Image3_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image3_CH0))
    #print("ch0 image 3 : ", len(Layer8_1st_Iter_Image3_CH0))

    Layer8_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x871AB800, End_Address=0x871B2000)
    Layer8_1st_Iter_Image4_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image4_CH0))
    #print("ch0 image 4 : ", len(Layer8_1st_Iter_Image4_CH0))

    Layer8_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x871B2000, End_Address=0x871B8800)
    Layer8_1st_Iter_Image5_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image5_CH0))
    #print("ch0 image 5 : ", len(Layer8_1st_Iter_Image5_CH0))

    Layer8_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x871B8800, End_Address=0x871BF000)
    Layer8_1st_Iter_Image6_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image6_CH0))
    #print("ch0 image 6 : ", len(Layer8_1st_Iter_Image6_CH0))

    Layer8_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x871BF000, End_Address=0x871C5800)
    Layer8_1st_Iter_Image7_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image7_CH0))
    #print("ch0 image 7 : ", len(Layer8_1st_Iter_Image7_CH0))

    Layer8_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x871C5800, End_Address=0x871CC000)
    Layer8_1st_Iter_Image8_CH0_256 = (process_input_lines(Layer8_1st_Iter_Image8_CH0))
    #print("ch0 image 8 : ", len(Layer8_1st_Iter_Image8_CH0))


    Layer8_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x97198000, End_Address=0x9719E800)
    Layer8_1st_Iter_Image1_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image1_CH1))   
    #print("ch1 image 1 : ", len(Layer8_1st_Iter_Image1_CH1))     

    Layer8_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x9719E800, End_Address=0x971A5000)
    Layer8_1st_Iter_Image2_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image2_CH1))
    #print("ch1 image 2 : ", len(Layer8_1st_Iter_Image2_CH1))

    Layer8_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x971A5000, End_Address=0x971AB800)
    Layer8_1st_Iter_Image3_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image3_CH1))
    #print("ch1 image 3 : ", len(Layer8_1st_Iter_Image3_CH1))

    Layer8_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x971AB800, End_Address=0x971B2000)
    Layer8_1st_Iter_Image4_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image4_CH1))
    #print("ch1 image 4 : ", len(Layer8_1st_Iter_Image4_CH1))

    Layer8_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x971B2000, End_Address=0x971B8800)
    Layer8_1st_Iter_Image5_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image5_CH1))
    #print("ch1 image 5 : ", len(Layer8_1st_Iter_Image5_CH1))

    Layer8_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x971B8800, End_Address=0x971BF000)
    Layer8_1st_Iter_Image6_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image6_CH1))
    #print("ch1 image 6 : ", len(Layer8_1st_Iter_Image6_CH1))

    Layer8_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x971BF000, End_Address=0x971C5800)
    Layer8_1st_Iter_Image7_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image7_CH1))
    #print("ch1 image 7 : ", len(Layer8_1st_Iter_Image7_CH1))

    Layer8_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x971C5800, End_Address=0x971CC000)
    Layer8_1st_Iter_Image8_CH1_256 = (process_input_lines(Layer8_1st_Iter_Image8_CH1))
    #print("ch1 image 8 : ", len(Layer8_1st_Iter_Image8_CH1))

    '''
    test_out = '1st_iter_result/Layer8_1st_Iter_Image1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = '1st_iter_result/Layer8_1st_Iter_Image8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Layer8_1st_Iter_Image8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''
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
    '''
    d = Device("0000:08:00.0")
    bar = d.bar[2]
    
    data_read = open("result/layer8_result_ch0.txt", mode="w+")
    i=0
    for i in range(0,int((0X71CC000-0X7198000)/4) ): 
        Read_Data = bar.read(0X7198000 + (i*4))
        data_read.write(str(Read_Data) + "\n")      

    data_read = open("result/layer8_result_ch1.txt", mode="w+")
    i=0
    for i in range(0,int((0X71CC000-0X7198000)/4) ): 
        Read_Data = bar.read(0X17198000 + (i*4))
        data_read.write(str(Read_Data) + "\n") 
    '''

    if Mode == "Training":
        Loss, Loss_Gradient = PostProcessing.PostProcessing()
        print(Loss)
        #print(Loss_Gradient)
        
        output_file1 = "result/loss.txt"
        with open(output_file1, mode="a") as output_file_1:
            output_file_1.write(str(Loss) + "\n")
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
    '''
    data_read = open("result/layer8_location_result.txt", mode="w+")
    i=0
    for i in range(0,int((0X82A8000-0X8274000)/4) ): 
        Read_Data = bar.read(0X8274000 + (i*4))
        data_read.write(str(Read_Data) + "\n")   
    '''
    # Backpropagation
    # if BN_Param_Converted:
    #     Backward_Const_List = PreProcessing.Backward_Const_Param_Converted_Func()           
    #     Average_Per_Channel_List = PreProcessing.Average_Per_Channel_Param_Converted_Func()   
            

    if YOLOv2_Hardware_Backward:
        # Weight_Backward_Layer8 for Soft2Hardware
        # if epoch == 0:
        Weight_Backward_Layer8 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 1024, Weight_List[8]+["0000"]*3072, ["0000"]*128, ["0000"]*128)
        # else :
        #     Weight_Backward_Layer8 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 1024, Weight_List[8], ["0000"]*128, ["0000"]*128)
        # print("Weight_Backward_Layer8: " + str(len(Weight_Backward_Layer8[0])))
        # print("Weight_Backward_Layer8: " + str(len(Weight_Backward_Layer8[1])))
        
        # # Weight_Backward_Layer7 for Soft2Hardware
        # Weight_Backward_Layer7 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], Backward_Const_List[7], Average_Per_Channel_List[7])
        # print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer7[0])))
        # print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer7[1])))
        
        # # Weight_Backward_Layer6 for Soft2Hardware
        # Weight_Backward_Layer6 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 512, Weight_List[6], Backward_Const_List[6], Average_Per_Channel_List[6])
        # print("Weight_Backward_Layer6: " + str(len(Weight_Backward_Layer6[0])))
        # print("Weight_Backward_Layer6: " + str(len(Weight_Backward_Layer6[1])))

        # # Weight_Backward_Layer5 for Soft2Hardware
        # Weight_Backward_Layer5 = Weight_Hardware_Backward_ReOrdering_OtherLayer(512, 256, Weight_List[5], Backward_Const_List[5], Average_Per_Channel_List[5])
        # print("Weight_Backward_Layer5: " + str(len(Weight_Backward_Layer5[0])))
        # print("Weight_Backward_Layer5: " + str(len(Weight_Backward_Layer5[1])))

        # # Weight_Backward_Layer4 for Soft2Hardware
        # Weight_Backward_Layer4 = Weight_Hardware_Backward_ReOrdering_OtherLayer(256, 128, Weight_List[4], Backward_Const_List[4], Average_Per_Channel_List[4])
        # print("Weight_Backward_Layer4: " + str(len(Weight_Backward_Layer4[0])))
        # print("Weight_Backward_Layer4: " + str(len(Weight_Backward_Layer4[1])))

        # # Weight_Backward_Layer3 for Soft2Hardware
        # Weight_Backward_Layer3 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 64, Weight_List[3], Backward_Const_List[3], Average_Per_Channel_List[3])
        # print("Weight_Backward_Layer3: " + str(len(Weight_Backward_Layer3[0])))
        # print("Weight_Backward_Layer3: " + str(len(Weight_Backward_Layer3[1])))

        # # Weight_Backward_Layer2 for Soft2Hardware
        # Weight_Backward_Layer2 = Weight_Hardware_Backward_ReOrdering_OtherLayer(64, 32, Weight_List[2], Backward_Const_List[2], Average_Per_Channel_List[2])
        # print("Weight_Backward_Layer2: " + str(len(Weight_Backward_Layer2[0])))
        # print("Weight_Backward_Layer2: " + str(len(Weight_Backward_Layer2[1])))

        # # Weight_Backward_Layer1 for Soft2Hardware
        # Weight_Backward_Layer1 = Weight_Hardware_Backward_ReOrdering_OtherLayer(32, 16, Weight_List[1], Backward_Const_List[1], Average_Per_Channel_List[1])
        # print("Weight_Backward_Layer1: " + str(len(Weight_Backward_Layer1[0])))
        # print("Weight_Backward_Layer1: " + str(len(Weight_Backward_Layer1[1])))

        # # Weight for Soft2Hardware
        # Weight_Backward_Layer0 = Weight_Hardware_Backward_ReOrdering_Layer0(16, 16, Weight_List[0], Backward_Const_List[0], Average_Per_Channel_List[0])
        # print("Weight_Layer0: " + str(len(Weight_Backward_Layer0[0])))
        # print("Weight_Layer0: " + str(len(Weight_Backward_Layer0[1])))

        '''
        # Total Weight for Backward: 
        Weight_Backward_CH0 = Weight_Backward_Layer0[0] + Weight_Backward_Layer1[0] + Weight_Backward_Layer2[0] + Weight_Backward_Layer3[0] + Weight_Backward_Layer4[0] + \
                        Weight_Backward_Layer5[0] + Weight_Backward_Layer6[0] + Weight_Backward_Layer7[0] + Weight_Backward_Layer8[0]
        Weight_Backward_CH1 = Weight_Backward_Layer0[1] + Weight_Backward_Layer6[1] + Weight_Backward_Layer2[1] + Weight_Backward_Layer3[1] + Weight_Backward_Layer4[1] + \
                        Weight_Backward_Layer5[1] + Weight_Backward_Layer6[1] + Weight_Backward_Layer7[1] + Weight_Backward_Layer8[1]
        
        # Break 256To32 and Flip the Data: 
        Weight_Backward_CH0 = Break_FlipHex_256To32(Weight_Backward_CH0)
        Weight_Backward_CH1 = Break_FlipHex_256To32(Weight_Backward_CH1)
        
        # Write Weight For Backward into DDR
        Write_DDR(Weight_Backward_CH0,Wr_Address=0x1200000)
        print("\t --> " + " Write Weight_Backward_CH0 Done!")
        Write_DDR(Weight_Backward_CH1,Wr_Address=0x11200000)
        print("\t --> " + " Write Weight_Backward_CH1 Done!")
        '''

        
        # Break 256To32 and Flip the Data: 
        Weight_Backward_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer8[0])
        Weight_Backward_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer8[1])

        # Write Weight For Backward into DDR
        Write_DDR(Weight_Backward_CH0,Wr_Address=0x81200000)
        print("\t --> " + " Write Weight_Backward_CH0 Done!")
        Write_DDR(Weight_Backward_CH1,Wr_Address=0x91200000)
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
        Loss_Grad1_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad1_layer8)
        # Loss_Grad2:
        Loss_Grad2_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient2_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad2_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad2_layer8) 
        # Loss_Grad3:  
        Loss_Grad3_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient3_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad3_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad3_layer8) 
        # Loss_Grad4:  
        Loss_Grad4_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient4_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad4_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad4_layer8) 
        # Loss_Grad5:
        Loss_Grad5_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient5_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad5_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad5_layer8) 
        # Loss_Grad6:
        Loss_Grad6_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient6_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad6_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad6_layer8) 
        # Loss_Grad7:  
        Loss_Grad7_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient7_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad7_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad7_layer8) 
        # Loss_Grad8:  
        Loss_Grad8_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient8_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad8_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad8_layer8)
        
        # Separate the DDR Channel: 
        Loss_Grad_layer8_CH0 =  Loss_Grad1_layer8[0] + Loss_Grad2_layer8[0] + Loss_Grad3_layer8[0] + Loss_Grad4_layer8[0] + \
                                Loss_Grad5_layer8[0] + Loss_Grad6_layer8[0] + Loss_Grad7_layer8[0] + Loss_Grad8_layer8[0]
        
        Loss_Grad_layer8_CH1 =  Loss_Grad1_layer8[1] + Loss_Grad2_layer8[1] + Loss_Grad3_layer8[1] + Loss_Grad4_layer8[1] + \
                                Loss_Grad5_layer8[1] + Loss_Grad6_layer8[1] + Loss_Grad7_layer8[1] + Loss_Grad8_layer8[1]
        

        # valid_values0 = [str(value) for value in Loss_Grad_layer8_CH0 if isinstance(value, (int, float))]
        # Loss_Grad_layer8_CH0_str = torch.tensor([float(value) for value in Loss_Grad_layer8_CH0], dtype=torch.float32)
        # Loss_Grad_layer8_CH1_str = torch.tensor([float(value) for value in Loss_Grad_layer8_CH1], dtype=torch.float32)

        # Loss_Grad_layer8_CH0_str = ' '.join(valid_values0)

        # print("Loss_Grad_layer8_CH0 : ", Loss_Grad_layer8_CH0)

        # valid_values1 = [str(value) for value in Loss_Grad_layer8_CH1 if isinstance(value, (int, float))]

        # Loss_Grad_layer8_CH1_str = ' '.join(valid_values1)
        

        # Write Loss Gradient to DDR:
        Write_DDR(Break_FlipHex_256To32(Loss_Grad_layer8_CH0), Wr_Address=0x882A8000)
        print("\t --> " + " Write Loss_Grad_layer8_CH0 Done!")
        Write_DDR(Break_FlipHex_256To32(Loss_Grad_layer8_CH1), Wr_Address=0x982A8000)
        print("\t --> " + " Write Loss_Grad_layer8_CH1 Done!") 


        d = Device("0000:08:00.0")
        bar = d.bar[0]
        
        resume()
        #print(irq_val)

        '''
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
        '''    

        # Bias Gradient Calculation
        Bias_Grad = torch.sum(Loss_Gradient, dim=(0, 2, 3))   

def Backward():
    layer8_start = time.time()
    #################################################
    #             Backward Layer 8 Start            #
    #################################################
    global Weight_Gradient_Layer0, Weight_Gradient_Layer1, Weight_Gradient_Layer2, Weight_Gradient_Layer3, Weight_Gradient_Layer4,\
    Weight_Gradient_Layer5, Weight_Gradient_Layer6, Weight_Gradient_Layer7, Weight_Gradient_Layer8
    # global Beta_Gradient_Layer0, Beta_Gradient_Layer1, Beta_Gradient_Layer2, Beta_Gradient_Layer3, Beta_Gradient_Layer4, Beta_Gradient_Layer5, Beta_Gradient_Layer6,\
    #     Beta_Gradient_Layer7
    global Gamma_Gradient_Layer0, Gamma_Gradient_Layer1, Gamma_Gradient_Layer2, Gamma_Gradient_Layer3, Gamma_Gradient_Layer4, Gamma_Gradient_Layer5,\
    Gamma_Gradient_Layer6, Gamma_Gradient_Layer7
    global dL_dgamma_7, dL_dbeta_7, dL_dgamma_6, dL_dbeta_6, dL_dgamma_5, dL_dbeta_5, dL_dgamma_4, dL_dbeta_4, dL_dgamma_3, dL_dbeta_3, dL_dgamma_2, dL_dbeta_2, \
        dL_dgamma_1, dL_dbeta_1, dL_dgamma_0, dL_dbeta_0
    
    global Gamma_Gradient, Beta_Gradient, Weight_Gradient
        
    # check Layer8 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer8_CH0 = Read_DDR(Rd_Address=0x86E58000,  End_Address=0x86E8C000)
    Output_Grad1_Layer8_CH0 = process_input_lines(Output_Grad1_Layer8_CH0)
    #print("Read Output_Grad1_Layer8_CH0")

    Output_Grad1_Layer8_CH1 = Read_DDR(Rd_Address=0x96E58000,  End_Address=0x96E8C000)
    Output_Grad1_Layer8_CH1 = process_input_lines(Output_Grad1_Layer8_CH1)
    #print("Read Output_Grad1_Layer8_CH1")
    
    Output_Grad2_Layer8_CH0 = Read_DDR(Rd_Address=0x86E8C000,  End_Address=0x86EC0000)
    Output_Grad2_Layer8_CH0 = process_input_lines(Output_Grad2_Layer8_CH0)
    #print("Read Output_Grad2_Layer8_CH0")

    Output_Grad2_Layer8_CH1 = Read_DDR(Rd_Address=0x96E8C000,  End_Address=0x96EC0000)
    Output_Grad2_Layer8_CH1 = process_input_lines(Output_Grad2_Layer8_CH1)
    #print("Read Output_Grad2_Layer8_CH1")

    Output_Grad3_Layer8_CH0 = Read_DDR(Rd_Address=0x86EC0000,  End_Address=0x86EF4000)
    Output_Grad3_Layer8_CH0 = process_input_lines(Output_Grad3_Layer8_CH0)
    #print("Read Output_Grad3_Layer8_CH0")

    Output_Grad3_Layer8_CH1 = Read_DDR(Rd_Address=0x96EC0000,  End_Address=0x96EF4000)
    Output_Grad3_Layer8_CH1 = process_input_lines(Output_Grad3_Layer8_CH1)
    #print("Read Output_Grad3_Layer8_CH1")

    Output_Grad4_Layer8_CH0 = Read_DDR(Rd_Address=0x86EF4000,  End_Address=0x86F28000)
    Output_Grad4_Layer8_CH0 = process_input_lines(Output_Grad4_Layer8_CH0)
    #print("Read Output_Grad4_Layer8_CH0")

    Output_Grad4_Layer8_CH1 = Read_DDR(Rd_Address=0x96EF4000,  End_Address=0x96F28000)
    Output_Grad4_Layer8_CH1 = process_input_lines(Output_Grad4_Layer8_CH1)
    #print("Read Output_Grad4_Layer8_CH1")

    Output_Grad5_Layer8_CH0 = Read_DDR(Rd_Address=0x86F28000,  End_Address=0x86F5C000)
    Output_Grad5_Layer8_CH0 = process_input_lines(Output_Grad5_Layer8_CH0)
    #print("Read Output_Grad5_Layer8_CH0")

    Output_Grad5_Layer8_CH1 = Read_DDR(Rd_Address=0x96F28000,  End_Address=0x96F5C000)
    Output_Grad5_Layer8_CH1 = process_input_lines(Output_Grad5_Layer8_CH1)
    #print("Read Output_Grad5_Layer8_CH1")

    Output_Grad6_Layer8_CH0 = Read_DDR(Rd_Address=0x86F5C000,  End_Address=0x86F90000)
    Output_Grad6_Layer8_CH0 = process_input_lines(Output_Grad6_Layer8_CH0)
    #print("Read Output_Grad6_Layer8_CH0")

    Output_Grad6_Layer8_CH1 = Read_DDR(Rd_Address=0x96F5C000,  End_Address=0x96F90000)
    Output_Grad6_Layer8_CH1 = process_input_lines(Output_Grad6_Layer8_CH1)
    #print("Read Output_Grad6_Layer8_CH1")

    Output_Grad7_Layer8_CH0 = Read_DDR(Rd_Address=0x86F90000,  End_Address=0x86FC4000)
    Output_Grad7_Layer8_CH0 = process_input_lines(Output_Grad7_Layer8_CH0)
    #print("Read Output_Grad7_Layer8_CH0")

    Output_Grad7_Layer8_CH1 = Read_DDR(Rd_Address=0x96F90000,  End_Address=0x96FC4000)
    Output_Grad7_Layer8_CH1 = process_input_lines(Output_Grad7_Layer8_CH1)
    #print("Read Output_Grad7_Layer8_CH1")

    Output_Grad8_Layer8_CH0 = Read_DDR(Rd_Address=0x86FC4000,  End_Address=0x86FF8000)
    Output_Grad8_Layer8_CH0 = process_input_lines(Output_Grad8_Layer8_CH0)
    #print("Read Output_Grad8_Layer8_CH0")

    Output_Grad8_Layer8_CH1 = Read_DDR(Rd_Address=0x96FC4000,  End_Address=0x96FF8000)
    Output_Grad8_Layer8_CH1 = process_input_lines(Output_Grad8_Layer8_CH1)    
    #print("Read Output_Grad8_Layer8_CH1")

    print("Convert Output Gradient")
    Output_Grad1_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer8_CH0, Output_Grad1_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad2_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer8_CH0, Output_Grad2_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad3_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer8_CH0, Output_Grad3_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad4_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer8_CH0, Output_Grad4_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad5_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer8_CH0, Output_Grad5_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad6_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer8_CH0, Output_Grad6_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad7_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer8_CH0, Output_Grad7_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grad8_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer8_CH0, Output_Grad8_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    Output_Grads_Layer8 = Output_Grad1_Layer8 + Output_Grad2_Layer8 + Output_Grad3_Layer8 + Output_Grad4_Layer8 + \
                            Output_Grad5_Layer8 + Output_Grad6_Layer8 + Output_Grad7_Layer8 + Output_Grad8_Layer8    
    Output_Grad_Layer8 = torch.tensor([float(value) for value in Output_Grads_Layer8], dtype=torch.float32).reshape(8, 1024, 13, 13)


    # BReLu Marking for Layer7
    ReLu_Marking1_Layer7_CH0 = Read_DDR(Rd_Address=0x880D4000,  End_Address=0x88108000)
    ReLu_Marking1_Layer7_CH0_256 = process_input_lines(ReLu_Marking1_Layer7_CH0)

    ReLu_Marking2_Layer7_CH0 = Read_DDR(Rd_Address=0x88108000,  End_Address=0x8813C000)
    ReLu_Marking2_Layer7_CH0_256 = process_input_lines(ReLu_Marking2_Layer7_CH0)

    ReLu_Marking3_Layer7_CH0 = Read_DDR(Rd_Address=0x8813C000,  End_Address=0x88170000)
    ReLu_Marking3_Layer7_CH0_256 = process_input_lines(ReLu_Marking3_Layer7_CH0)

    ReLu_Marking4_Layer7_CH0 = Read_DDR(Rd_Address=0x88170000,  End_Address=0x881A4000)
    ReLu_Marking4_Layer7_CH0_256 = process_input_lines(ReLu_Marking4_Layer7_CH0)

    ReLu_Marking5_Layer7_CH0 = Read_DDR(Rd_Address=0x881A4000,  End_Address=0x881D8000)
    ReLu_Marking5_Layer7_CH0_256 = process_input_lines(ReLu_Marking5_Layer7_CH0)

    ReLu_Marking6_Layer7_CH0 = Read_DDR(Rd_Address=0x881D8000,  End_Address=0x8820C000)
    ReLu_Marking6_Layer7_CH0_256 = process_input_lines(ReLu_Marking6_Layer7_CH0)

    ReLu_Marking7_Layer7_CH0 = Read_DDR(Rd_Address=0x8820C000,  End_Address=0x88240000)
    ReLu_Marking7_Layer7_CH0_256 = process_input_lines(ReLu_Marking7_Layer7_CH0)

    ReLu_Marking8_Layer7_CH0 = Read_DDR(Rd_Address=0x88240000,  End_Address=0x88274000)
    ReLu_Marking8_Layer7_CH0_256 = process_input_lines(ReLu_Marking8_Layer7_CH0)

    ReLu_Marking1_Layer7_CH1 = Read_DDR(Rd_Address=0x980D4000,  End_Address=0x98108000)
    ReLu_Marking1_Layer7_CH1_256 = process_input_lines(ReLu_Marking1_Layer7_CH1)

    ReLu_Marking2_Layer7_CH1 = Read_DDR(Rd_Address=0x98108000,  End_Address=0x9813C000)
    ReLu_Marking2_Layer7_CH1_256 = process_input_lines(ReLu_Marking2_Layer7_CH1)

    ReLu_Marking3_Layer7_CH1 = Read_DDR(Rd_Address=0x9813C000,  End_Address=0x98170000)
    ReLu_Marking3_Layer7_CH1_256 = process_input_lines(ReLu_Marking3_Layer7_CH1)

    ReLu_Marking4_Layer7_CH1 = Read_DDR(Rd_Address=0x98170000,  End_Address=0x981A4000)
    ReLu_Marking4_Layer7_CH1_256 = process_input_lines(ReLu_Marking4_Layer7_CH1)

    ReLu_Marking5_Layer7_CH1 = Read_DDR(Rd_Address=0x981A4000,  End_Address=0x981D8000)
    ReLu_Marking5_Layer7_CH1_256 = process_input_lines(ReLu_Marking5_Layer7_CH1)

    ReLu_Marking6_Layer7_CH1 = Read_DDR(Rd_Address=0x981D8000,  End_Address=0x9820C000)
    ReLu_Marking6_Layer7_CH1_256 = process_input_lines(ReLu_Marking6_Layer7_CH1)

    ReLu_Marking7_Layer7_CH1 = Read_DDR(Rd_Address=0x9820C000,  End_Address=0x98240000)
    ReLu_Marking7_Layer7_CH1_256 = process_input_lines(ReLu_Marking7_Layer7_CH1)

    ReLu_Marking8_Layer7_CH1 = Read_DDR(Rd_Address=0x98240000,  End_Address=0x98274000)
    ReLu_Marking8_Layer7_CH1_256 = process_input_lines(ReLu_Marking8_Layer7_CH1)

    # ReLu Reordering
    ReLu_Marking1_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer7_CH0_256, ReLu_Marking1_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking2_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer7_CH0_256, ReLu_Marking2_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking3_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer7_CH0_256, ReLu_Marking3_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking4_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer7_CH0_256, ReLu_Marking4_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking5_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer7_CH0_256, ReLu_Marking5_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking6_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer7_CH0_256, ReLu_Marking6_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking7_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer7_CH0_256, ReLu_Marking7_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking8_Layer7 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer7_CH0_256, ReLu_Marking8_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)


    ReLu_Marking_Layer7 = ReLu_Marking1_Layer7 + ReLu_Marking2_Layer7 + ReLu_Marking3_Layer7 + ReLu_Marking4_Layer7 + ReLu_Marking5_Layer7 + \
                            ReLu_Marking6_Layer7 + ReLu_Marking7_Layer7 + ReLu_Marking8_Layer7
    
    ReLu_Marking_Layer7 = torch.tensor([float(value) for value in ReLu_Marking_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)


    # BReLu Calculate
    # Output_Grad_layer8_input = torch.tensor(Output_Grad_Layer8, dtype=torch.float32).reshape(8,1024,13,13)
    # Layer7_Location = torch.tensor(ReLu_Marking_Layer7, dtype=torch.float32).reshape(8,1024,13,13)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer7)
    grad_relu_output = backward_active(Output_Grad_Layer8, relu_mask)
    #grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_7, dL_dbeta_7, avg_pc_7, backward_const_7 = backward_LightNorm(grad_relu_output, layer7_cache)

    test_out = 'result/ReLu_Marking_Layer7.txt'
    with open(test_out, 'w+') as test_output:
        for item in ReLu_Marking_Layer7:
            test_output.write(str(item) + "\n")
    test_output.close()

    test_out = 'result/grad_relu_output.txt'
    with open(test_out, 'w+') as test_output:
        for item in grad_relu_output:
            test_output.write(str(item) + "\n")
    test_output.close()

    test_out = 'result/Output_Grad_Layer8.txt'
    with open(test_out, 'w+') as test_output:
        for item in Output_Grad_Layer8:
            test_output.write(str(item) + "\n")
    test_output.close()

    # avg_pc_7 = avg_pc_7.squeeze()
    # backward_const_7 = backward_const_7.squeeze()

    avg_pc_7, backward_const_7 = Mean_Var_Dec2Bfloat(avg_pc_7, backward_const_7, Exponent_Bits, Mantissa_Bits)

    # Weight Gradient
    Weight_Gradient1_Layer8_CH0 = Read_DDR(Rd_Address=0x882DC000,  End_Address=0x882FC000)
    Weight_Gradient1_Layer8_CH0_256 = process_input_lines(Weight_Gradient1_Layer8_CH0)
    #print("Weight_Gradient1_Layer8_CH0 : ", len(Weight_Gradient1_Layer8_CH0))   

    Weight_Gradient2_Layer8_CH0 = Read_DDR(Rd_Address=0x882FC000,  End_Address=0x8831C000)
    Weight_Gradient2_Layer8_CH0_256 = process_input_lines(Weight_Gradient2_Layer8_CH0)
    #print("Weight_Gradient2_Layer8_CH0 : ", len(Weight_Gradient2_Layer8_CH0))    

    Weight_Gradient3_Layer8_CH0 = Read_DDR(Rd_Address=0x8831C000,  End_Address=0x8833C000)
    Weight_Gradient3_Layer8_CH0_256 = process_input_lines(Weight_Gradient3_Layer8_CH0)
    #print("Weight_Gradient3_Layer8_CH0 : ", len(Weight_Gradient3_Layer8_CH0)) 

    Weight_Gradient4_Layer8_CH0 = Read_DDR(Rd_Address=0x8833C000,  End_Address=0x8835C000)
    Weight_Gradient4_Layer8_CH0_256 = process_input_lines(Weight_Gradient4_Layer8_CH0)
    #print("Weight_Gradient4_Layer8_CH0 : ", len(Weight_Gradient4_Layer8_CH0)) 

    Weight_Gradient5_Layer8_CH0 = Read_DDR(Rd_Address=0x8835C000,  End_Address=0X8837C000)
    Weight_Gradient5_Layer8_CH0_256 = process_input_lines(Weight_Gradient5_Layer8_CH0)
    #print("Weight_Gradient5_Layer8_CH0 : ", len(Weight_Gradient5_Layer8_CH0)) 

    Weight_Gradient6_Layer8_CH0 = Read_DDR(Rd_Address=0X8837C000,  End_Address=0x8839C000)
    Weight_Gradient6_Layer8_CH0_256 = process_input_lines(Weight_Gradient6_Layer8_CH0)
    #print("Weight_Gradient6_Layer8_CH0 : ", len(Weight_Gradient6_Layer8_CH0)) 

    Weight_Gradient7_Layer8_CH0 = Read_DDR(Rd_Address=0x8839C000,  End_Address=0x883BC000)
    Weight_Gradient7_Layer8_CH0_256 = process_input_lines(Weight_Gradient7_Layer8_CH0)
    #print("Weight_Gradient7_Layer8_CH0 : ", len(Weight_Gradient7_Layer8_CH0)) 

    Weight_Gradient8_Layer8_CH0 = Read_DDR(Rd_Address=0x883BC000,  End_Address=0x883DC000)
    Weight_Gradient8_Layer8_CH0_256 = process_input_lines(Weight_Gradient8_Layer8_CH0)
    #print("Weight_Gradient8_Layer8_CH0 : ", len(Weight_Gradient8_Layer8_CH0)) 


    Weight_Gradient1_Layer8_CH1 = Read_DDR(Rd_Address=0x982DC000,  End_Address=0x982FC000)
    Weight_Gradient1_Layer8_CH1_256 = process_input_lines(Weight_Gradient1_Layer8_CH1)
    #print("Weight_Gradient1_Layer8_CH1 : ", len(Weight_Gradient1_Layer8_CH1))   

    Weight_Gradient2_Layer8_CH1 = Read_DDR(Rd_Address=0x982FC000,  End_Address=0x9831C000)
    Weight_Gradient2_Layer8_CH1_256 = process_input_lines(Weight_Gradient2_Layer8_CH1)
    #print("Weight_Gradient2_Layer8_CH1 : ", len(Weight_Gradient2_Layer8_CH1))    

    Weight_Gradient3_Layer8_CH1 = Read_DDR(Rd_Address=0x9831C000,  End_Address=0x9833C000)
    Weight_Gradient3_Layer8_CH1_256 = process_input_lines(Weight_Gradient3_Layer8_CH1)
    #print("Weight_Gradient3_Layer8_CH1 : ", len(Weight_Gradient3_Layer8_CH1)) 

    Weight_Gradient4_Layer8_CH1 = Read_DDR(Rd_Address=0x9833C000,  End_Address=0x9835C000)
    Weight_Gradient4_Layer8_CH1_256 = process_input_lines(Weight_Gradient4_Layer8_CH1)
    #print("Weight_Gradient4_Layer8_CH1 : ", len(Weight_Gradient4_Layer8_CH1)) 

    Weight_Gradient5_Layer8_CH1 = Read_DDR(Rd_Address=0x9835C000,  End_Address=0x9837C000)
    Weight_Gradient5_Layer8_CH1_256 = process_input_lines(Weight_Gradient5_Layer8_CH1)
    #print("Weight_Gradient5_Layer8_CH1 : ", len(Weight_Gradient5_Layer8_CH1)) 

    Weight_Gradient6_Layer8_CH1 = Read_DDR(Rd_Address=0x9837C000,  End_Address=0x9839C000)
    Weight_Gradient6_Layer8_CH1_256 = process_input_lines(Weight_Gradient6_Layer8_CH1)
    #print("Weight_Gradient6_Layer8_CH1 : ", len(Weight_Gradient6_Layer8_CH1)) 

    Weight_Gradient7_Layer8_CH1 = Read_DDR(Rd_Address=0x9839C000,  End_Address=0x983BC000)
    Weight_Gradient7_Layer8_CH1_256 = process_input_lines(Weight_Gradient7_Layer8_CH1)
    #print("Weight_Gradient7_Layer8_CH1 : ", len(Weight_Gradient7_Layer8_CH1)) 

    Weight_Gradient8_Layer8_CH1 = Read_DDR(Rd_Address=0x983BC000,  End_Address=0x983DC000)
    Weight_Gradient8_Layer8_CH1_256 = process_input_lines(Weight_Gradient8_Layer8_CH1)
    #print("Weight_Gradient8_Layer8_CH1 : ", len(Weight_Gradient8_Layer8_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer8_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer8_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer8_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer8_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

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
    #Weight_Gradient_Layer8 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer8)]   
    Weight_Gradient_Layer8 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer8)]


    Weight_Gradient_Layer8 = torch.tensor([float(value) for value in Weight_Gradient_Layer8], dtype=torch.float32).reshape(125, 1024, 1, 1)   

    # Backward_Const_List[7] = backward_const_7
    # Average_Per_Channel_List[7] = avg_pc_7

    # Weight_Backward_Layer7 for Soft2Hardware
    Weight_Backward_Layer7 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], backward_const_7, avg_pc_7)

    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer7_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer7[0])
    Weight_Backward_Layer7_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer7[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer7_CH0,Wr_Address=0x81340000)
    Write_DDR(Weight_Backward_Layer7_CH1,Wr_Address=0x91340000)

    layer8_end = time.time()
    process_time = layer8_end - layer8_start
    print("Layer8 Process Time : ", process_time)

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    ##print(irq_val)



    #################################################
    #             Backward Layer 7 Start            #
    #################################################
    layer7_start = time.time()
    # check Layer7 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer7_CH0 = Read_DDR(Rd_Address=0x86B18000,  End_Address=0x86B4C000)
    Output_Grad1_Layer7_CH0 = process_input_lines(Output_Grad1_Layer7_CH0)

    Output_Grad1_Layer7_CH1 = Read_DDR(Rd_Address=0x96B18000,  End_Address=0x96B4C000)
    Output_Grad1_Layer7_CH1 = process_input_lines(Output_Grad1_Layer7_CH1)

    Output_Grad2_Layer7_CH0 = Read_DDR(Rd_Address=0x86B4C000,  End_Address=0x86B80000)
    Output_Grad2_Layer7_CH0 = process_input_lines(Output_Grad2_Layer7_CH0)

    Output_Grad2_Layer7_CH1 = Read_DDR(Rd_Address=0x96B4C000,  End_Address=0x96B80000)
    Output_Grad2_Layer7_CH1 = process_input_lines(Output_Grad2_Layer7_CH1)

    Output_Grad3_Layer7_CH0 = Read_DDR(Rd_Address=0x86B80000,  End_Address=0x86BB4000)
    Output_Grad3_Layer7_CH0 = process_input_lines(Output_Grad3_Layer7_CH0)

    Output_Grad3_Layer7_CH1 = Read_DDR(Rd_Address=0x96B80000,  End_Address=0x96BB4000)
    Output_Grad3_Layer7_CH1 = process_input_lines(Output_Grad3_Layer7_CH1)

    Output_Grad4_Layer7_CH0 = Read_DDR(Rd_Address=0x86BB4000,  End_Address=0x86BE8000)
    Output_Grad4_Layer7_CH0 = process_input_lines(Output_Grad4_Layer7_CH0)

    Output_Grad4_Layer7_CH1 = Read_DDR(Rd_Address=0x96BB4000,  End_Address=0x96BE8000)
    Output_Grad4_Layer7_CH1 = process_input_lines(Output_Grad4_Layer7_CH1)

    Output_Grad5_Layer7_CH0 = Read_DDR(Rd_Address=0x86BE8000,  End_Address=0x86C1C000)
    Output_Grad5_Layer7_CH0 = process_input_lines(Output_Grad5_Layer7_CH0)

    Output_Grad5_Layer7_CH1 = Read_DDR(Rd_Address=0x96BE8000,  End_Address=0x96C1C000)
    Output_Grad5_Layer7_CH1 = process_input_lines(Output_Grad5_Layer7_CH1)

    Output_Grad6_Layer7_CH0 = Read_DDR(Rd_Address=0x86C1C000,  End_Address=0x86C50000)
    Output_Grad6_Layer7_CH0 = process_input_lines(Output_Grad6_Layer7_CH0)

    Output_Grad6_Layer7_CH1 = Read_DDR(Rd_Address=0x96C1C000,  End_Address=0x96C50000)
    Output_Grad6_Layer7_CH1 = process_input_lines(Output_Grad6_Layer7_CH1)

    Output_Grad7_Layer7_CH0 = Read_DDR(Rd_Address=0x86C50000,  End_Address=0x86C84000)
    Output_Grad7_Layer7_CH0 = process_input_lines(Output_Grad7_Layer7_CH0)

    Output_Grad7_Layer7_CH1 = Read_DDR(Rd_Address=0x96C50000,  End_Address=0x96C84000)
    Output_Grad7_Layer7_CH1 = process_input_lines(Output_Grad7_Layer7_CH1)

    Output_Grad8_Layer7_CH0 = Read_DDR(Rd_Address=0x86C84000,  End_Address=0x86CB8000)
    Output_Grad8_Layer7_CH0 = process_input_lines(Output_Grad8_Layer7_CH0)

    Output_Grad8_Layer7_CH1 = Read_DDR(Rd_Address=0x96C84000,  End_Address=0x96CB8000)
    Output_Grad8_Layer7_CH1 = process_input_lines(Output_Grad8_Layer7_CH1)

    print("Convert")
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

    # BReLu Marking
    ReLu_Marking1_Layer6_CH0 = Read_DDR(Rd_Address=0x87F34000,  End_Address=0x87F68000)
    ReLu_Marking1_Layer6_CH0_256 = process_input_lines(ReLu_Marking1_Layer6_CH0)

    ReLu_Marking2_Layer6_CH0 = Read_DDR(Rd_Address=0x87F68000,  End_Address=0x87F9C000)
    ReLu_Marking2_Layer6_CH0_256 = process_input_lines(ReLu_Marking2_Layer6_CH0)

    ReLu_Marking3_Layer6_CH0 = Read_DDR(Rd_Address=0x87F9C000,  End_Address=0x87FD0000)
    ReLu_Marking3_Layer6_CH0_256 = process_input_lines(ReLu_Marking3_Layer6_CH0)

    ReLu_Marking4_Layer6_CH0 = Read_DDR(Rd_Address=0x87FD0000,  End_Address=0x88004000)
    ReLu_Marking4_Layer6_CH0_256 = process_input_lines(ReLu_Marking4_Layer6_CH0)

    ReLu_Marking5_Layer6_CH0 = Read_DDR(Rd_Address=0x88004000,  End_Address=0x88038000)
    ReLu_Marking5_Layer6_CH0_256 = process_input_lines(ReLu_Marking5_Layer6_CH0)

    ReLu_Marking6_Layer6_CH0 = Read_DDR(Rd_Address=0x88038000,  End_Address=0x8806C000)
    ReLu_Marking6_Layer6_CH0_256 = process_input_lines(ReLu_Marking6_Layer6_CH0)

    ReLu_Marking7_Layer6_CH0 = Read_DDR(Rd_Address=0x8806C000,  End_Address=0x880A0000)
    ReLu_Marking7_Layer6_CH0_256 = process_input_lines(ReLu_Marking7_Layer6_CH0)

    ReLu_Marking8_Layer6_CH0 = Read_DDR(Rd_Address=0x880A0000,  End_Address=0x880D4000)
    ReLu_Marking8_Layer6_CH0_256 = process_input_lines(ReLu_Marking8_Layer6_CH0)

    ReLu_Marking1_Layer6_CH1 = Read_DDR(Rd_Address=0x97F34000,  End_Address=0x97F68000)
    ReLu_Marking1_Layer6_CH1_256 = process_input_lines(ReLu_Marking1_Layer6_CH1)

    ReLu_Marking2_Layer6_CH1 = Read_DDR(Rd_Address=0x97F68000,  End_Address=0x97F9C000)
    ReLu_Marking2_Layer6_CH1_256 = process_input_lines(ReLu_Marking2_Layer6_CH1)

    ReLu_Marking3_Layer6_CH1 = Read_DDR(Rd_Address=0x97F9C000,  End_Address=0x97FD0000)
    ReLu_Marking3_Layer6_CH1_256 = process_input_lines(ReLu_Marking3_Layer6_CH1)

    ReLu_Marking4_Layer6_CH1 = Read_DDR(Rd_Address=0x97FD0000,  End_Address=0x98004000)
    ReLu_Marking4_Layer6_CH1_256 = process_input_lines(ReLu_Marking4_Layer6_CH1)

    ReLu_Marking5_Layer6_CH1 = Read_DDR(Rd_Address=0x98004000,  End_Address=0x98038000)
    ReLu_Marking5_Layer6_CH1_256 = process_input_lines(ReLu_Marking5_Layer6_CH1)

    ReLu_Marking6_Layer6_CH1 = Read_DDR(Rd_Address=0x98038000,  End_Address=0x9806C000)
    ReLu_Marking6_Layer6_CH1_256 = process_input_lines(ReLu_Marking6_Layer6_CH1)

    ReLu_Marking7_Layer6_CH1 = Read_DDR(Rd_Address=0x9806C000,  End_Address=0x980A0000)
    ReLu_Marking7_Layer6_CH1_256 = process_input_lines(ReLu_Marking7_Layer6_CH1)

    ReLu_Marking8_Layer6_CH1 = Read_DDR(Rd_Address=0x980A0000,  End_Address=0x980D4000)
    ReLu_Marking8_Layer6_CH1_256 = process_input_lines(ReLu_Marking8_Layer6_CH1)

    # ReLu Reordering
    ReLu_Marking1_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer6_CH0_256, ReLu_Marking1_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking2_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer6_CH0_256, ReLu_Marking2_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking3_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer6_CH0_256, ReLu_Marking3_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking4_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer6_CH0_256, ReLu_Marking4_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking5_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer6_CH0_256, ReLu_Marking5_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking6_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer6_CH0_256, ReLu_Marking6_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking7_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer6_CH0_256, ReLu_Marking7_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    ReLu_Marking8_Layer6 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer6_CH0_256, ReLu_Marking8_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)


    ReLu_Marking_Layer6 = ReLu_Marking1_Layer6 + ReLu_Marking2_Layer6 + ReLu_Marking3_Layer6 + ReLu_Marking4_Layer6 + ReLu_Marking5_Layer6 + \
                            ReLu_Marking6_Layer6 + ReLu_Marking7_Layer6 + ReLu_Marking8_Layer6
    
    ReLu_Marking_Layer6 = torch.tensor([float(value) for value in ReLu_Marking_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)

    # BReLu Calculate
    # Output_Grad_layer7_input = torch.tensor(Output_Grad_Layer7, dtype=torch.float32).reshape(8,1024,13,13)
    # Layer6_Location = torch.tensor(ReLu_Marking_Layer6, dtype=torch.float32).reshape(8,1024,13,13)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer6)
    grad_relu_output = backward_active(Output_Grad_Layer7, relu_mask)
    #grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_6, dL_dbeta_6, avg_pc_6, backward_const_6 = backward_LightNorm(grad_relu_output, layer6_cache)

    # avg_pc_6 = avg_pc_6.squeeze()
    # backward_const_6 = backward_const_6.squeeze()

    avg_pc_6, backward_const_6 = Mean_Var_Dec2Bfloat(avg_pc_6, backward_const_6, Exponent_Bits, Mantissa_Bits)

    # Weight_Backward_Layer6 for Soft2Hardware
    Weight_Backward_Layer6 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 512, Weight_List[6], backward_const_6, avg_pc_6)
    #print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer6[0])))
    #print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer6[1])))

    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer6_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer6[0])
    Weight_Backward_Layer6_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer6[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer6_CH0,Wr_Address=0x81D40000)
    Write_DDR(Weight_Backward_Layer6_CH1,Wr_Address=0x91D40000)

    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer7 = (Output_Grad_Layer7).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer7_CH0 = Read_DDR(Rd_Address=0x883DC000,  End_Address=0x88CDC000)
    Weight_Gradient1_Layer7_CH0_256 = process_input_lines(Weight_Gradient1_Layer7_CH0)
    #print("Weight_Gradient1_Layer7_CH0 : ", len(Weight_Gradient1_Layer7_CH0))   

    Weight_Gradient2_Layer7_CH0 = Read_DDR(Rd_Address=0x88CDC000,  End_Address=0x895DC000)
    Weight_Gradient2_Layer7_CH0_256 = process_input_lines(Weight_Gradient2_Layer7_CH0)
    #print("Weight_Gradient2_Layer7_CH0 : ", len(Weight_Gradient2_Layer7_CH0))    

    Weight_Gradient3_Layer7_CH0 = Read_DDR(Rd_Address=0x895DC000,  End_Address=0x89EDC000)
    Weight_Gradient3_Layer7_CH0_256 = process_input_lines(Weight_Gradient3_Layer7_CH0)
    #print("Weight_Gradient3_Layer7_CH0 : ", len(Weight_Gradient3_Layer7_CH0)) 

    Weight_Gradient4_Layer7_CH0 = Read_DDR(Rd_Address=0x89EDC000,  End_Address=0x8A7DC000)
    Weight_Gradient4_Layer7_CH0_256 = process_input_lines(Weight_Gradient4_Layer7_CH0)
    #print("Weight_Gradient4_Layer7_CH0 : ", len(Weight_Gradient4_Layer7_CH0)) 

    Weight_Gradient5_Layer7_CH0 = Read_DDR(Rd_Address=0x8A7DC000,  End_Address=0x8B0DC000)
    Weight_Gradient5_Layer7_CH0_256 = process_input_lines(Weight_Gradient5_Layer7_CH0)
    #print("Weight_Gradient5_Layer7_CH0 : ", len(Weight_Gradient5_Layer7_CH0)) 

    Weight_Gradient6_Layer7_CH0 = Read_DDR(Rd_Address=0x8B0DC000,  End_Address=0x8B9DC000)
    Weight_Gradient6_Layer7_CH0_256 = process_input_lines(Weight_Gradient6_Layer7_CH0)
    #print("Weight_Gradient6_Layer7_CH0 : ", len(Weight_Gradient6_Layer7_CH0)) 

    Weight_Gradient7_Layer7_CH0 = Read_DDR(Rd_Address=0x8B9DC000,  End_Address=0x8C2DC000)
    Weight_Gradient7_Layer7_CH0_256 = process_input_lines(Weight_Gradient7_Layer7_CH0)
    #print("Weight_Gradient7_Layer7_CH0 : ", len(Weight_Gradient7_Layer7_CH0)) 

    Weight_Gradient8_Layer7_CH0 = Read_DDR(Rd_Address=0x8C2DC000,  End_Address=0x8CBDC000)
    Weight_Gradient8_Layer7_CH0_256 = process_input_lines(Weight_Gradient8_Layer7_CH0)
    #print("Weight_Gradient8_Layer7_CH0 : ", len(Weight_Gradient8_Layer7_CH0)) 

    Weight_Gradient1_Layer7_CH1 = Read_DDR(Rd_Address=0x983DC000,  End_Address=0x98CDC000)
    Weight_Gradient1_Layer7_CH1_256 = process_input_lines(Weight_Gradient1_Layer7_CH1)
    #print("Weight_Gradient1_Layer7_CH1 : ", len(Weight_Gradient1_Layer7_CH1)) 

    Weight_Gradient2_Layer7_CH1 = Read_DDR(Rd_Address=0x98CDC000,  End_Address=0x995DC000)
    Weight_Gradient2_Layer7_CH1_256 = process_input_lines(Weight_Gradient2_Layer7_CH1)
    #print("Weight_Gradient2_Layer7_CH1 : ", len(Weight_Gradient2_Layer7_CH1)) 

    Weight_Gradient3_Layer7_CH1 = Read_DDR(Rd_Address=0x995DC000,  End_Address=0x99EDC000)
    Weight_Gradient3_Layer7_CH1_256 = process_input_lines(Weight_Gradient3_Layer7_CH1)
    #print("Weight_Gradient3_Layer7_CH1 : ", len(Weight_Gradient3_Layer7_CH1)) 

    Weight_Gradient4_Layer7_CH1 = Read_DDR(Rd_Address=0x99EDC000,  End_Address=0x9A7DC000)
    Weight_Gradient4_Layer7_CH1_256 = process_input_lines(Weight_Gradient4_Layer7_CH1)
    #print("Weight_Gradient4_Layer7_CH1 : ", len(Weight_Gradient4_Layer7_CH1)) 

    Weight_Gradient5_Layer7_CH1 = Read_DDR(Rd_Address=0x9A7DC000,  End_Address=0x9B0DC000)
    Weight_Gradient5_Layer7_CH1_256 = process_input_lines(Weight_Gradient5_Layer7_CH1)
    #print("Weight_Gradient5_Layer7_CH1 : ", len(Weight_Gradient5_Layer7_CH1)) 

    Weight_Gradient6_Layer7_CH1 = Read_DDR(Rd_Address=0x9B0DC000,  End_Address=0x9B9DC000)
    Weight_Gradient6_Layer7_CH1_256 = process_input_lines(Weight_Gradient6_Layer7_CH1)
    #print("Weight_Gradient6_Layer7_CH1 : ", len(Weight_Gradient6_Layer7_CH1)) 

    Weight_Gradient7_Layer7_CH1 = Read_DDR(Rd_Address=0x9B9DC000,  End_Address=0x9C2DC000)
    Weight_Gradient7_Layer7_CH1_256 = process_input_lines(Weight_Gradient7_Layer7_CH1)
    #print("Weight_Gradient7_Layer7_CH1 : ", len(Weight_Gradient7_Layer7_CH1)) 

    Weight_Gradient8_Layer7_CH1 = Read_DDR(Rd_Address=0x9C2DC000,  End_Address=0x9CBDC000)
    Weight_Gradient8_Layer7_CH1_256 = process_input_lines(Weight_Gradient8_Layer7_CH1)
    #print("Weight_Gradient8_Layer7_CH1 : ", len(Weight_Gradient8_Layer7_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer7_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer7_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer7_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer7_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

    print("Convert")
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
    
    Weight_Gradient_Layer7 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer7)]   
    Weight_Gradient_Layer7 = torch.tensor([float(value) for value in Weight_Gradient_Layer7], dtype=torch.float32).reshape(1024, 1024, 3, 3)  

    layer7_end = time.time()
    process_time = layer7_end - layer7_start
    print("Layer7 Process Time : ", process_time)  

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    ##print(irq_val) 


    #################################################
    #             Backward Layer 6 Start            #
    #################################################
    layer6_start = time.time()
    # check Layer7 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer6_CH0 = Read_DDR(Rd_Address=0x86978000,  End_Address=0x86992000)
    Output_Grad1_Layer6_CH0 = process_input_lines(Output_Grad1_Layer6_CH0)

    Output_Grad1_Layer6_CH1 = Read_DDR(Rd_Address=0x96978000,  End_Address=0x96992000)
    Output_Grad1_Layer6_CH1 = process_input_lines(Output_Grad1_Layer6_CH1)

    Output_Grad2_Layer6_CH0 = Read_DDR(Rd_Address=0x86992000,  End_Address=0x869AC000)
    Output_Grad2_Layer6_CH0 = process_input_lines(Output_Grad2_Layer6_CH0)

    Output_Grad2_Layer6_CH1 = Read_DDR(Rd_Address=0x96992000,  End_Address=0x969AC000)
    Output_Grad2_Layer6_CH1 = process_input_lines(Output_Grad2_Layer6_CH1)

    Output_Grad3_Layer6_CH0 = Read_DDR(Rd_Address=0x869AC000,  End_Address=0x869C6000)
    Output_Grad3_Layer6_CH0 = process_input_lines(Output_Grad3_Layer6_CH0)

    Output_Grad3_Layer6_CH1 = Read_DDR(Rd_Address=0x969AC000,  End_Address=0x969C6000)
    Output_Grad3_Layer6_CH1 = process_input_lines(Output_Grad3_Layer6_CH1)

    Output_Grad4_Layer6_CH0 = Read_DDR(Rd_Address=0x869C6000,  End_Address=0x869E0000)
    Output_Grad4_Layer6_CH0 = process_input_lines(Output_Grad4_Layer6_CH0)

    Output_Grad4_Layer6_CH1 = Read_DDR(Rd_Address=0x969C6000,  End_Address=0x969E0000)
    Output_Grad4_Layer6_CH1 = process_input_lines(Output_Grad4_Layer6_CH1)

    Output_Grad5_Layer6_CH0 = Read_DDR(Rd_Address=0x869E0000,  End_Address=0x869FA000)
    Output_Grad5_Layer6_CH0 = process_input_lines(Output_Grad5_Layer6_CH0)

    Output_Grad5_Layer6_CH1 = Read_DDR(Rd_Address=0x969E0000,  End_Address=0x969FA000)
    Output_Grad5_Layer6_CH1 = process_input_lines(Output_Grad5_Layer6_CH1)

    Output_Grad6_Layer6_CH0 = Read_DDR(Rd_Address=0x869FA000,  End_Address=0x86A14000)
    Output_Grad6_Layer6_CH0 = process_input_lines(Output_Grad6_Layer6_CH0)

    Output_Grad6_Layer6_CH1 = Read_DDR(Rd_Address=0x969FA000,  End_Address=0x96A14000)
    Output_Grad6_Layer6_CH1 = process_input_lines(Output_Grad6_Layer6_CH1)

    Output_Grad7_Layer6_CH0 = Read_DDR(Rd_Address=0x86A14000,  End_Address=0x86A2E000)
    Output_Grad7_Layer6_CH0 = process_input_lines(Output_Grad7_Layer6_CH0)

    Output_Grad7_Layer6_CH1 = Read_DDR(Rd_Address=0x96A14000,  End_Address=0x96A2E000)
    Output_Grad7_Layer6_CH1 = process_input_lines(Output_Grad7_Layer6_CH1)

    Output_Grad8_Layer6_CH0 = Read_DDR(Rd_Address=0x86A2E000,  End_Address=0x86A48000)
    Output_Grad8_Layer6_CH0 = process_input_lines(Output_Grad8_Layer6_CH0)

    Output_Grad8_Layer6_CH1 = Read_DDR(Rd_Address=0x96A2E000,  End_Address=0x96A48000)
    Output_Grad8_Layer6_CH1 = process_input_lines(Output_Grad8_Layer6_CH1)


    Output_Grad1_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer6_CH0, Output_Grad1_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad2_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer6_CH0, Output_Grad2_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad3_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer6_CH0, Output_Grad3_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad4_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer6_CH0, Output_Grad4_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad5_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer6_CH0, Output_Grad5_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad6_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer6_CH0, Output_Grad6_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad7_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer6_CH0, Output_Grad7_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grad8_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer6_CH0, Output_Grad8_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    Output_Grads_Layer6 = Output_Grad1_Layer6 + Output_Grad2_Layer6 + Output_Grad3_Layer6 + Output_Grad4_Layer6 + \
                            Output_Grad5_Layer6 + Output_Grad6_Layer6 + Output_Grad7_Layer6 + Output_Grad8_Layer6    
    Output_Grad_Layer6 = torch.tensor([float(value) for value in Output_Grads_Layer6], dtype=torch.float32).reshape(8, 512, 13, 13)


    # BReLu Marking
    ReLu_Marking1_Layer5_CH0 = Read_DDR(Rd_Address=0x87E64000,  End_Address=0x87E7E000)
    ReLu_Marking1_Layer5_CH0_256 = process_input_lines(ReLu_Marking1_Layer5_CH0)

    ReLu_Marking2_Layer5_CH0 = Read_DDR(Rd_Address=0x87E7E000,  End_Address=0x87E98000)
    ReLu_Marking2_Layer5_CH0_256 = process_input_lines(ReLu_Marking2_Layer5_CH0)

    ReLu_Marking3_Layer5_CH0 = Read_DDR(Rd_Address=0x87E98000,  End_Address=0x87EB2000)
    ReLu_Marking3_Layer5_CH0_256 = process_input_lines(ReLu_Marking3_Layer5_CH0)

    ReLu_Marking4_Layer5_CH0 = Read_DDR(Rd_Address=0x87EB2000,  End_Address=0x87ECC000)
    ReLu_Marking4_Layer5_CH0_256 = process_input_lines(ReLu_Marking4_Layer5_CH0)

    ReLu_Marking5_Layer5_CH0 = Read_DDR(Rd_Address=0x87ECC000,  End_Address=0x87EE6000)
    ReLu_Marking5_Layer5_CH0_256 = process_input_lines(ReLu_Marking5_Layer5_CH0)

    ReLu_Marking6_Layer5_CH0 = Read_DDR(Rd_Address=0x87EE6000,  End_Address=0x87F00000)
    ReLu_Marking6_Layer5_CH0_256 = process_input_lines(ReLu_Marking6_Layer5_CH0)

    ReLu_Marking7_Layer5_CH0 = Read_DDR(Rd_Address=0x87F00000,  End_Address=0x87F1A000)
    ReLu_Marking7_Layer5_CH0_256 = process_input_lines(ReLu_Marking7_Layer5_CH0)

    ReLu_Marking8_Layer5_CH0 = Read_DDR(Rd_Address=0x87F1A000,  End_Address=0x87F34000)
    ReLu_Marking8_Layer5_CH0_256 = process_input_lines(ReLu_Marking8_Layer5_CH0)

    ReLu_Marking1_Layer5_CH1 = Read_DDR(Rd_Address=0x97E64000,  End_Address=0x97E7E000)
    ReLu_Marking1_Layer5_CH1_256 = process_input_lines(ReLu_Marking1_Layer5_CH1)

    ReLu_Marking2_Layer5_CH1 = Read_DDR(Rd_Address=0x97E7E000,  End_Address=0x97E98000)
    ReLu_Marking2_Layer5_CH1_256 = process_input_lines(ReLu_Marking2_Layer5_CH1)

    ReLu_Marking3_Layer5_CH1 = Read_DDR(Rd_Address=0x97E98000,  End_Address=0x97EB2000)
    ReLu_Marking3_Layer5_CH1_256 = process_input_lines(ReLu_Marking3_Layer5_CH1)

    ReLu_Marking4_Layer5_CH1 = Read_DDR(Rd_Address=0x97EB2000,  End_Address=0x97ECC000)
    ReLu_Marking4_Layer5_CH1_256 = process_input_lines(ReLu_Marking4_Layer5_CH1)

    ReLu_Marking5_Layer5_CH1 = Read_DDR(Rd_Address=0x97ECC000,  End_Address=0x97EE6000)
    ReLu_Marking5_Layer5_CH1_256 = process_input_lines(ReLu_Marking5_Layer5_CH1)

    ReLu_Marking6_Layer5_CH1 = Read_DDR(Rd_Address=0x97EE6000,  End_Address=0x97F00000)
    ReLu_Marking6_Layer5_CH1_256 = process_input_lines(ReLu_Marking6_Layer5_CH1)

    ReLu_Marking7_Layer5_CH1 = Read_DDR(Rd_Address=0x97F00000,  End_Address=0x97F1A000)
    ReLu_Marking7_Layer5_CH1_256 = process_input_lines(ReLu_Marking7_Layer5_CH1)

    ReLu_Marking8_Layer5_CH1 = Read_DDR(Rd_Address=0x97F1A000,  End_Address=0x97F34000)
    ReLu_Marking8_Layer5_CH1_256 = process_input_lines(ReLu_Marking8_Layer5_CH1)

    # ReLu Reordering
    ReLu_Marking1_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer5_CH0_256, ReLu_Marking1_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking2_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer5_CH0_256, ReLu_Marking2_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking3_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer5_CH0_256, ReLu_Marking3_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking4_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer5_CH0_256, ReLu_Marking4_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking5_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer5_CH0_256, ReLu_Marking5_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking6_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer5_CH0_256, ReLu_Marking6_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking7_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer5_CH0_256, ReLu_Marking7_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    ReLu_Marking8_Layer5 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer5_CH0_256, ReLu_Marking8_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)


    ReLu_Marking_Layer5 = ReLu_Marking1_Layer5 + ReLu_Marking2_Layer5 + ReLu_Marking3_Layer5 + ReLu_Marking4_Layer5 + ReLu_Marking5_Layer5 + \
                            ReLu_Marking6_Layer5 + ReLu_Marking7_Layer5 + ReLu_Marking8_Layer5
    
    ReLu_Marking_Layer5 = torch.tensor([float(value) for value in ReLu_Marking_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)


    # BReLu Calculate
    # Output_Grad_layer6_input = torch.tensor(Output_Grad_Layer6, dtype=torch.float32).reshape(8,512,13,13)
    # Layer5_Location = torch.tensor(ReLu_Marking_Layer5, dtype=torch.float32).reshape(8,512,13,13)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer5)
    grad_relu_output = backward_active(Output_Grad_Layer6, relu_mask)
    #grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_5, dL_dbeta_5, avg_pc_5, backward_const_5 = backward_LightNorm(grad_relu_output, layer5_cache)

    # avg_pc_5 = avg_pc_5.squeeze()
    # backward_const_5 = backward_const_5.squeeze()

    avg_pc_5, backward_const_5 = Mean_Var_Dec2Bfloat(avg_pc_5, backward_const_5, Exponent_Bits, Mantissa_Bits)

    # Weight_Backward_Layer5 for Soft2Hardware
    Weight_Backward_Layer5 = Weight_Hardware_Backward_ReOrdering_OtherLayer(512, 256, Weight_List[5], backward_const_5, avg_pc_5)

    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer5_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer5[0])
    Weight_Backward_Layer5_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer5[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer5_CH0,Wr_Address=0x82240000)
    Write_DDR(Weight_Backward_Layer5_CH1,Wr_Address=0x92240000)

    
    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer6 = (Output_Grad_Layer6).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer6_CH0 = Read_DDR(Rd_Address=0x8CBDC000,  End_Address=0x8D05C000)
    Weight_Gradient1_Layer6_CH0_256 = process_input_lines(Weight_Gradient1_Layer6_CH0)
    #print("Weight_Gradient1_Layer6_CH0 : ", len(Weight_Gradient1_Layer6_CH0))   

    Weight_Gradient2_Layer6_CH0 = Read_DDR(Rd_Address=0x8D05C000,  End_Address=0x8D4DC000)
    Weight_Gradient2_Layer6_CH0_256 = process_input_lines(Weight_Gradient2_Layer6_CH0)
    #print("Weight_Gradient2_Layer6_CH0 : ", len(Weight_Gradient2_Layer6_CH0))    

    Weight_Gradient3_Layer6_CH0 = Read_DDR(Rd_Address=0x8D4DC000,  End_Address=0x8D95C000)
    Weight_Gradient3_Layer6_CH0_256 = process_input_lines(Weight_Gradient3_Layer6_CH0)
    #print("Weight_Gradient3_Layer6_CH0 : ", len(Weight_Gradient3_Layer6_CH0)) 

    Weight_Gradient4_Layer6_CH0 = Read_DDR(Rd_Address=0x8D95C000,  End_Address=0x8DDDC000)
    Weight_Gradient4_Layer6_CH0_256 = process_input_lines(Weight_Gradient4_Layer6_CH0)
    #print("Weight_Gradient4_Layer6_CH0 : ", len(Weight_Gradient4_Layer6_CH0)) 

    Weight_Gradient5_Layer6_CH0 = Read_DDR(Rd_Address=0x8DDDC000,  End_Address=0x8E25C000)
    Weight_Gradient5_Layer6_CH0_256 = process_input_lines(Weight_Gradient5_Layer6_CH0)
    #print("Weight_Gradient5_Layer6_CH0 : ", len(Weight_Gradient5_Layer6_CH0)) 

    Weight_Gradient6_Layer6_CH0 = Read_DDR(Rd_Address=0x8E25C000,  End_Address=0x8E6DC000)
    Weight_Gradient6_Layer6_CH0_256 = process_input_lines(Weight_Gradient6_Layer6_CH0)
    #print("Weight_Gradient6_Layer6_CH0 : ", len(Weight_Gradient6_Layer6_CH0)) 

    Weight_Gradient7_Layer6_CH0 = Read_DDR(Rd_Address=0x8E6DC000,  End_Address=0x8EB5C000)
    Weight_Gradient7_Layer6_CH0_256 = process_input_lines(Weight_Gradient7_Layer6_CH0)
    #print("Weight_Gradient7_Layer6_CH0 : ", len(Weight_Gradient7_Layer6_CH0)) 

    Weight_Gradient8_Layer6_CH0 = Read_DDR(Rd_Address=0x8EB5C000,  End_Address=0x8EFDC000)
    Weight_Gradient8_Layer6_CH0_256 = process_input_lines(Weight_Gradient8_Layer6_CH0)
    #print("Weight_Gradient8_Layer6_CH0 : ", len(Weight_Gradient8_Layer6_CH0)) 

    Weight_Gradient1_Layer6_CH1 = Read_DDR(Rd_Address=0x9CBDC000,  End_Address=0x9D05C000)
    Weight_Gradient1_Layer6_CH1_256 = process_input_lines(Weight_Gradient1_Layer6_CH1)
    #print("Weight_Gradient1_Layer6_CH1 : ", len(Weight_Gradient1_Layer6_CH1)) 

    Weight_Gradient2_Layer6_CH1 = Read_DDR(Rd_Address=0x9D05C000,  End_Address=0x9D4DC000)
    Weight_Gradient2_Layer6_CH1_256 = process_input_lines(Weight_Gradient2_Layer6_CH1)
    #print("Weight_Gradient2_Layer6_CH1 : ", len(Weight_Gradient2_Layer6_CH1)) 

    Weight_Gradient3_Layer6_CH1 = Read_DDR(Rd_Address=0x9D4DC000,  End_Address=0x9D95C000)
    Weight_Gradient3_Layer6_CH1_256 = process_input_lines(Weight_Gradient3_Layer6_CH1)
    #print("Weight_Gradient3_Layer6_CH1 : ", len(Weight_Gradient3_Layer6_CH1)) 

    Weight_Gradient4_Layer6_CH1 = Read_DDR(Rd_Address=0x9D95C000,  End_Address=0x9DDDC000)
    Weight_Gradient4_Layer6_CH1_256 = process_input_lines(Weight_Gradient4_Layer6_CH1)
    #print("Weight_Gradient4_Layer6_CH1 : ", len(Weight_Gradient4_Layer6_CH1)) 

    Weight_Gradient5_Layer6_CH1 = Read_DDR(Rd_Address=0x9DDDC000,  End_Address=0x9E25C000)
    Weight_Gradient5_Layer6_CH1_256 = process_input_lines(Weight_Gradient5_Layer6_CH1)
    #print("Weight_Gradient5_Layer6_CH1 : ", len(Weight_Gradient5_Layer6_CH1)) 

    Weight_Gradient6_Layer6_CH1 = Read_DDR(Rd_Address=0x9E25C000,  End_Address=0x9E6DC000)
    Weight_Gradient6_Layer6_CH1_256 = process_input_lines(Weight_Gradient6_Layer6_CH1)
    #print("Weight_Gradient6_Layer6_CH1 : ", len(Weight_Gradient6_Layer6_CH1)) 

    Weight_Gradient7_Layer6_CH1 = Read_DDR(Rd_Address=0x9E6DC000,  End_Address=0x9EB5C000)
    Weight_Gradient7_Layer6_CH1_256 = process_input_lines(Weight_Gradient7_Layer6_CH1)
    #print("Weight_Gradient7_Layer6_CH1 : ", len(Weight_Gradient7_Layer6_CH1)) 

    Weight_Gradient8_Layer6_CH1 = Read_DDR(Rd_Address=0x9EB5C000,  End_Address=0x9EFDC000)
    Weight_Gradient8_Layer6_CH1_256 = process_input_lines(Weight_Gradient8_Layer6_CH1)
    #print("Weight_Gradient8_Layer6_CH1 : ", len(Weight_Gradient8_Layer6_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer6_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer6_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer6_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer6_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

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
    Weight_Gradient_Layer6 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer6)]   
    Weight_Gradient_Layer6 = torch.tensor([float(value) for value in Weight_Gradient_Layer6], dtype=torch.float32).reshape(1024, 512, 3, 3)  

    layer6_end = time.time()
    process_time = layer6_end - layer6_start
    print("Layer6 Process Time : ", process_time)

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    #print(irq_val)  

    #################################################
    #             Backward Layer 5 Start            #
    #################################################
    layer5_start = time.time()
    # check Layer5 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer5_CH0 = Read_DDR(Rd_Address=0x86770000,  End_Address=0x8677D000)
    Output_Grad1_Layer5_CH0 = process_input_lines(Output_Grad1_Layer5_CH0)
    #print("Read Output_Grad1_Layer5_CH0")

    Output_Grad1_Layer5_CH1 = Read_DDR(Rd_Address=0x96770000,  End_Address=0x9677D000)
    Output_Grad1_Layer5_CH1 = process_input_lines(Output_Grad1_Layer5_CH1)
    #print("Read Output_Grad1_Layer5_CH1")

    Output_Grad2_Layer5_CH0 = Read_DDR(Rd_Address=0x8677D000,  End_Address=0x8678A000)
    Output_Grad2_Layer5_CH0 = process_input_lines(Output_Grad2_Layer5_CH0)
    #print("Read Output_Grad2_Layer5_CH0")

    Output_Grad2_Layer5_CH1 = Read_DDR(Rd_Address=0x9677D000,  End_Address=0x9678A000)
    Output_Grad2_Layer5_CH1 = process_input_lines(Output_Grad2_Layer5_CH1)
    #print("Read Output_Grad2_Layer5_CH1")

    Output_Grad3_Layer5_CH0 = Read_DDR(Rd_Address=0x8678A000,  End_Address=0X86797000)
    Output_Grad3_Layer5_CH0 = process_input_lines(Output_Grad3_Layer5_CH0)
    #print("Read Output_Grad3_Layer5_CH0")

    Output_Grad3_Layer5_CH1 = Read_DDR(Rd_Address=0x9678A000,  End_Address=0x96797000)
    Output_Grad3_Layer5_CH1 = process_input_lines(Output_Grad3_Layer5_CH1)
    #print("Read Output_Grad3_Layer5_CH1")

    Output_Grad4_Layer5_CH0 = Read_DDR(Rd_Address=0x86797000,  End_Address=0x867A4000)
    Output_Grad4_Layer5_CH0 = process_input_lines(Output_Grad4_Layer5_CH0)
    #print("Read Output_Grad4_Layer5_CH0")

    Output_Grad4_Layer5_CH1 = Read_DDR(Rd_Address=0x96797000,  End_Address=0x967A4000)
    Output_Grad4_Layer5_CH1 = process_input_lines(Output_Grad4_Layer5_CH1)
    #print("Read Output_Grad4_Layer5_CH1")

    Output_Grad5_Layer5_CH0 = Read_DDR(Rd_Address=0x867A4000,  End_Address=0x867B1000)
    Output_Grad5_Layer5_CH0 = process_input_lines(Output_Grad5_Layer5_CH0)
    #print("Read Output_Grad5_Layer5_CH0")

    Output_Grad5_Layer5_CH1 = Read_DDR(Rd_Address=0x967A4000,  End_Address=0x967B1000)
    Output_Grad5_Layer5_CH1 = process_input_lines(Output_Grad5_Layer5_CH1)
    #print("Read Output_Grad5_Layer5_CH1")

    Output_Grad6_Layer5_CH0 = Read_DDR(Rd_Address=0x867B1000,  End_Address=0x867BE000)
    Output_Grad6_Layer5_CH0 = process_input_lines(Output_Grad6_Layer5_CH0)
    #print("Read Output_Grad6_Layer5_CH0")

    Output_Grad6_Layer5_CH1 = Read_DDR(Rd_Address=0x967B1000,  End_Address=0x967BE000)
    Output_Grad6_Layer5_CH1 = process_input_lines(Output_Grad6_Layer5_CH1)
    #print("Read Output_Grad6_Layer5_CH1")

    Output_Grad7_Layer5_CH0 = Read_DDR(Rd_Address=0x867BE000,  End_Address=0x867CB000)
    Output_Grad7_Layer5_CH0 = process_input_lines(Output_Grad7_Layer5_CH0)
    #print("Read Output_Grad7_Layer5_CH0")

    Output_Grad7_Layer5_CH1 = Read_DDR(Rd_Address=0x967BE000,  End_Address=0x967CB000)
    Output_Grad7_Layer5_CH1 = process_input_lines(Output_Grad7_Layer5_CH1)
    #print("Read Output_Grad7_Layer5_CH1")

    Output_Grad8_Layer5_CH0 = Read_DDR(Rd_Address=0x867CB000,  End_Address=0x867D8000)
    Output_Grad8_Layer5_CH0 = process_input_lines(Output_Grad8_Layer5_CH0)
    #print("Read Output_Grad8_Layer5_CH0")

    Output_Grad8_Layer5_CH1 = Read_DDR(Rd_Address=0x967CB000,  End_Address=0x967D8000)
    Output_Grad8_Layer5_CH1 = process_input_lines(Output_Grad8_Layer5_CH1)
    #print("Read Output_Grad8_Layer5_CH1")

    print("Convert Output Gradient")
    Output_Grad1_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer5_CH0, Output_Grad1_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad2_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer5_CH0, Output_Grad2_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad3_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer5_CH0, Output_Grad3_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad4_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer5_CH0, Output_Grad4_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad5_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer5_CH0, Output_Grad5_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad6_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer5_CH0, Output_Grad6_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad7_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer5_CH0, Output_Grad7_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grad8_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer5_CH0, Output_Grad8_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    Output_Grads_Layer5 = Output_Grad1_Layer5 + Output_Grad2_Layer5 + Output_Grad3_Layer5 + Output_Grad4_Layer5 + \
                            Output_Grad5_Layer5 + Output_Grad6_Layer5 + Output_Grad7_Layer5 + Output_Grad8_Layer5    
    Output_Grad_Layer5 = torch.tensor([float(value) for value in Output_Grads_Layer5], dtype=torch.float32).reshape(8, 256, 13, 13)

    # ReLU
    ReLu_Marking1_Layer4_CH0 = Read_DDR(Rd_Address=0x87DFC000,  End_Address=0x87E09000)
    ReLu_Marking1_Layer4_CH0_256 = process_input_lines(ReLu_Marking1_Layer4_CH0)

    ReLu_Marking2_Layer4_CH0 = Read_DDR(Rd_Address=0x87E09000,  End_Address=0x87E16000)
    ReLu_Marking2_Layer4_CH0_256 = process_input_lines(ReLu_Marking2_Layer4_CH0)

    ReLu_Marking3_Layer4_CH0 = Read_DDR(Rd_Address=0x87E16000,  End_Address=0x87E23000)
    ReLu_Marking3_Layer4_CH0_256 = process_input_lines(ReLu_Marking3_Layer4_CH0)

    ReLu_Marking4_Layer4_CH0 = Read_DDR(Rd_Address=0x87E23000,  End_Address=0x87E30000)
    ReLu_Marking4_Layer4_CH0_256 = process_input_lines(ReLu_Marking4_Layer4_CH0)

    ReLu_Marking5_Layer4_CH0 = Read_DDR(Rd_Address=0x87E30000,  End_Address=0x87E3D000)
    ReLu_Marking5_Layer4_CH0_256 = process_input_lines(ReLu_Marking5_Layer4_CH0)

    ReLu_Marking6_Layer4_CH0 = Read_DDR(Rd_Address=0x87E3D000,  End_Address=0x87E4A000)
    ReLu_Marking6_Layer4_CH0_256 = process_input_lines(ReLu_Marking6_Layer4_CH0)

    ReLu_Marking7_Layer4_CH0 = Read_DDR(Rd_Address=0x87E4A000,  End_Address=0x87E57000)
    ReLu_Marking7_Layer4_CH0_256 = process_input_lines(ReLu_Marking7_Layer4_CH0)

    ReLu_Marking8_Layer4_CH0 = Read_DDR(Rd_Address=0x87E57000,  End_Address=0x87E64000)
    ReLu_Marking8_Layer4_CH0_256 = process_input_lines(ReLu_Marking8_Layer4_CH0)

    ReLu_Marking1_Layer4_CH1 = Read_DDR(Rd_Address=0x97DFC000,  End_Address=0x97E09000)
    ReLu_Marking1_Layer4_CH1_256 = process_input_lines(ReLu_Marking1_Layer4_CH1)

    ReLu_Marking2_Layer4_CH1 = Read_DDR(Rd_Address=0x97E09000,  End_Address=0x97E16000)
    ReLu_Marking2_Layer4_CH1_256 = process_input_lines(ReLu_Marking2_Layer4_CH1)

    ReLu_Marking3_Layer4_CH1 = Read_DDR(Rd_Address=0x97E16000,  End_Address=0x97E23000)
    ReLu_Marking3_Layer4_CH1_256 = process_input_lines(ReLu_Marking3_Layer4_CH1)

    ReLu_Marking4_Layer4_CH1 = Read_DDR(Rd_Address=0x97E23000,  End_Address=0x97E30000)
    ReLu_Marking4_Layer4_CH1_256 = process_input_lines(ReLu_Marking4_Layer4_CH1)

    ReLu_Marking5_Layer4_CH1 = Read_DDR(Rd_Address=0x97E30000,  End_Address=0x97E3D000)
    ReLu_Marking5_Layer4_CH1_256 = process_input_lines(ReLu_Marking5_Layer4_CH1)

    ReLu_Marking6_Layer4_CH1 = Read_DDR(Rd_Address=0x97E3D000,  End_Address=0x97E4A000)
    ReLu_Marking6_Layer4_CH1_256 = process_input_lines(ReLu_Marking6_Layer4_CH1)

    ReLu_Marking7_Layer4_CH1 = Read_DDR(Rd_Address=0x97E4A000,  End_Address=0x97E57000)
    ReLu_Marking7_Layer4_CH1_256 = process_input_lines(ReLu_Marking7_Layer4_CH1)

    ReLu_Marking8_Layer4_CH1 = Read_DDR(Rd_Address=0x97E57000,  End_Address=0x97E64000)
    ReLu_Marking8_Layer4_CH1_256 = process_input_lines(ReLu_Marking8_Layer4_CH1)

    # ReLu Reordering
    print("Convert ReLu Masking")
    ReLu_Marking1_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer4_CH0_256, ReLu_Marking1_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking2_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer4_CH0_256, ReLu_Marking2_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking3_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer4_CH0_256, ReLu_Marking3_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking4_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer4_CH0_256, ReLu_Marking4_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking5_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer4_CH0_256, ReLu_Marking5_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking6_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer4_CH0_256, ReLu_Marking6_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking7_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer4_CH0_256, ReLu_Marking7_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    ReLu_Marking8_Layer4 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer4_CH0_256, ReLu_Marking8_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)


    ReLu_Marking_Layer4 = ReLu_Marking1_Layer4 + ReLu_Marking2_Layer4 + ReLu_Marking3_Layer4 + ReLu_Marking4_Layer4 + ReLu_Marking5_Layer4 + \
                            ReLu_Marking6_Layer4 + ReLu_Marking7_Layer4 + ReLu_Marking8_Layer4
    
    ReLu_Marking_Layer4 = torch.tensor([float(value) for value in ReLu_Marking_Layer4], dtype=torch.float32).reshape(8, 256, 13, 13)

    # BReLu Calculate
    # Output_Grad_layer5_input = torch.tensor(Output_Grad_Layer5, dtype=torch.float32).reshape(8,256,13,13)
    # Layer4_Location = torch.tensor(ReLu_Marking_Layer4, dtype=torch.float32).reshape(8,256,13,13)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer4)
    grad_relu_output = backward_active(Output_Grad_Layer5, relu_mask)
    grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_4, dL_dbeta_4, avg_pc_4, backward_const_4 = backward_LightNorm(grad_maxpool_output, layer4_cache)

    # avg_pc_4 = avg_pc_4.squeeze()
    # backward_const_4 = backward_const_4.squeeze()

    avg_pc_4, backward_const_4 = Mean_Var_Dec2Bfloat(avg_pc_4, backward_const_4, Exponent_Bits, Mantissa_Bits)


    # Weight_Backward_Layer4 for Soft2Hardware
    Weight_Backward_Layer4 = Weight_Hardware_Backward_ReOrdering_OtherLayer(256, 128, Weight_List[4], backward_const_4, avg_pc_4)


    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer4_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer4[0])
    Weight_Backward_Layer4_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer4[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer4_CH0,Wr_Address=0x82380000)
    Write_DDR(Weight_Backward_Layer4_CH1,Wr_Address=0x92380000)

    
    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer5 = (Output_Grad_Layer5).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer5_CH0 = Read_DDR(Rd_Address=0x8EFDC000,  End_Address=0x8F0FC000)
    Weight_Gradient1_Layer5_CH0_256 = process_input_lines(Weight_Gradient1_Layer5_CH0)
    #print("Weight_Gradient1_Layer5_CH0 : ", len(Weight_Gradient1_Layer5_CH0))   

    Weight_Gradient2_Layer5_CH0 = Read_DDR(Rd_Address=0x8F0FC000,  End_Address=0x8F21C000)
    Weight_Gradient2_Layer5_CH0_256 = process_input_lines(Weight_Gradient2_Layer5_CH0)
    #print("Weight_Gradient2_Layer5_CH0 : ", len(Weight_Gradient2_Layer5_CH0))    

    Weight_Gradient3_Layer5_CH0 = Read_DDR(Rd_Address=0x8F21C000,  End_Address=0x8F33C000)
    Weight_Gradient3_Layer5_CH0_256 = process_input_lines(Weight_Gradient3_Layer5_CH0)
    #print("Weight_Gradient3_Layer5_CH0 : ", len(Weight_Gradient3_Layer5_CH0)) 

    Weight_Gradient4_Layer5_CH0 = Read_DDR(Rd_Address=0x8F33C000,  End_Address=0x8F45C000)
    Weight_Gradient4_Layer5_CH0_256 = process_input_lines(Weight_Gradient4_Layer5_CH0)
    #print("Weight_Gradient4_Layer5_CH0 : ", len(Weight_Gradient4_Layer5_CH0)) 

    Weight_Gradient5_Layer5_CH0 = Read_DDR(Rd_Address=0x8F45C000,  End_Address=0x8F57C000)
    Weight_Gradient5_Layer5_CH0_256 = process_input_lines(Weight_Gradient5_Layer5_CH0)
    #print("Weight_Gradient5_Layer5_CH0 : ", len(Weight_Gradient5_Layer5_CH0)) 

    Weight_Gradient6_Layer5_CH0 = Read_DDR(Rd_Address=0x8F57C000,  End_Address=0x8F69C000)
    Weight_Gradient6_Layer5_CH0_256 = process_input_lines(Weight_Gradient6_Layer5_CH0)
    #print("Weight_Gradient6_Layer5_CH0 : ", len(Weight_Gradient6_Layer5_CH0)) 

    Weight_Gradient7_Layer5_CH0 = Read_DDR(Rd_Address=0x8F69C000,  End_Address=0x8F7BC000)
    Weight_Gradient7_Layer5_CH0_256 = process_input_lines(Weight_Gradient7_Layer5_CH0)
    #print("Weight_Gradient7_Layer5_CH0 : ", len(Weight_Gradient7_Layer5_CH0)) 

    Weight_Gradient8_Layer5_CH0 = Read_DDR(Rd_Address=0x8F7BC000,  End_Address=0x8F8DC000)
    Weight_Gradient8_Layer5_CH0_256 = process_input_lines(Weight_Gradient8_Layer5_CH0)
    #print("Weight_Gradient8_Layer5_CH0 : ", len(Weight_Gradient8_Layer5_CH0)) 

    Weight_Gradient1_Layer5_CH1 = Read_DDR(Rd_Address=0x9EFDC000,  End_Address=0x9F0FC000)
    Weight_Gradient1_Layer5_CH1_256 = process_input_lines(Weight_Gradient1_Layer5_CH1)
    #print("Weight_Gradient1_Layer5_CH1 : ", len(Weight_Gradient1_Layer5_CH1)) 

    Weight_Gradient2_Layer5_CH1 = Read_DDR(Rd_Address=0x9F0FC000,  End_Address=0x9F21C000)
    Weight_Gradient2_Layer5_CH1_256 = process_input_lines(Weight_Gradient2_Layer5_CH1)
    #print("Weight_Gradient2_Layer5_CH1 : ", len(Weight_Gradient2_Layer5_CH1)) 

    Weight_Gradient3_Layer5_CH1 = Read_DDR(Rd_Address=0x9F21C000,  End_Address=0x9F33C000)
    Weight_Gradient3_Layer5_CH1_256 = process_input_lines(Weight_Gradient3_Layer5_CH1)
    #print("Weight_Gradient3_Layer5_CH1 : ", len(Weight_Gradient3_Layer5_CH1)) 

    Weight_Gradient4_Layer5_CH1 = Read_DDR(Rd_Address=0x9F33C000,  End_Address=0x9F45C000)
    Weight_Gradient4_Layer5_CH1_256 = process_input_lines(Weight_Gradient4_Layer5_CH1)
    #print("Weight_Gradient4_Layer5_CH1 : ", len(Weight_Gradient4_Layer5_CH1)) 

    Weight_Gradient5_Layer5_CH1 = Read_DDR(Rd_Address=0x9F45C000,  End_Address=0x9F57C000)
    Weight_Gradient5_Layer5_CH1_256 = process_input_lines(Weight_Gradient5_Layer5_CH1)
    #print("Weight_Gradient5_Layer5_CH1 : ", len(Weight_Gradient5_Layer5_CH1)) 

    Weight_Gradient6_Layer5_CH1 = Read_DDR(Rd_Address=0x9F57C000,  End_Address=0x9F69C000)
    Weight_Gradient6_Layer5_CH1_256 = process_input_lines(Weight_Gradient6_Layer5_CH1)
    #print("Weight_Gradient6_Layer5_CH1 : ", len(Weight_Gradient6_Layer5_CH1)) 

    Weight_Gradient7_Layer5_CH1 = Read_DDR(Rd_Address=0x9F69C000,  End_Address=0x9F7BC000)
    Weight_Gradient7_Layer5_CH1_256 = process_input_lines(Weight_Gradient7_Layer5_CH1)
    #print("Weight_Gradient7_Layer5_CH1 : ", len(Weight_Gradient7_Layer5_CH1)) 

    Weight_Gradient8_Layer5_CH1 = Read_DDR(Rd_Address=0x9F7BC000,  End_Address=0x9F8DC000)
    Weight_Gradient8_Layer5_CH1_256 = process_input_lines(Weight_Gradient8_Layer5_CH1)
    #print("Weight_Gradient8_Layer5_CH1 : ", len(Weight_Gradient8_Layer5_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer5_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer5_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer5_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer5_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

    print("Convert Weight Gradient")
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
    Weight_Gradient_Layer5 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer5)]   
    Weight_Gradient_Layer5 = torch.tensor([float(value) for value in Weight_Gradient_Layer5], dtype=torch.float32).reshape(512, 256, 3, 3)  

    layer5_end = time.time()
    process_time = layer5_end - layer5_start
    print("Layer5 Process Time : ", process_time)

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    #print(irq_val)    

    #################################################
    #             Backward Layer 4 Start            #
    #################################################
    # check Layer4 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer4_CH0 = Read_DDR(Rd_Address=0x86360000,  End_Address=0x8637A000)
    Output_Grad1_Layer4_CH0 = process_input_lines(Output_Grad1_Layer4_CH0)
    #print("Read Output_Grad1_Layer4_CH0")

    Output_Grad1_Layer4_CH1 = Read_DDR(Rd_Address=0x96360000,  End_Address=0x9637A000)
    Output_Grad1_Layer4_CH1 = process_input_lines(Output_Grad1_Layer4_CH1)
    #print("Read Output_Grad1_Layer4_CH1")

    Output_Grad2_Layer4_CH0 = Read_DDR(Rd_Address=0x8637A000,  End_Address=0x86394000)
    Output_Grad2_Layer4_CH0 = process_input_lines(Output_Grad2_Layer4_CH0)
    #print("Read Output_Grad2_Layer4_CH0")

    Output_Grad2_Layer4_CH1 = Read_DDR(Rd_Address=0x9637A000,  End_Address=0x96394000)
    Output_Grad2_Layer4_CH1 = process_input_lines(Output_Grad2_Layer4_CH1)
    #print("Read Output_Grad2_Layer4_CH1")

    Output_Grad3_Layer4_CH0 = Read_DDR(Rd_Address=0x86394000,  End_Address=0x863AE000)
    Output_Grad3_Layer4_CH0 = process_input_lines(Output_Grad3_Layer4_CH0)
    #print("Read Output_Grad3_Layer4_CH0")

    Output_Grad3_Layer4_CH1 = Read_DDR(Rd_Address=0x96394000,  End_Address=0x963AE000)
    Output_Grad3_Layer4_CH1 = process_input_lines(Output_Grad3_Layer4_CH1)
    #print("Read Output_Grad3_Layer4_CH1")

    Output_Grad4_Layer4_CH0 = Read_DDR(Rd_Address=0x863AE000,  End_Address=0x863C8000)
    Output_Grad4_Layer4_CH0 = process_input_lines(Output_Grad4_Layer4_CH0)
    #print("Read Output_Grad4_Layer4_CH0")

    Output_Grad4_Layer4_CH1 = Read_DDR(Rd_Address=0x963AE000,  End_Address=0x963C8000)
    Output_Grad4_Layer4_CH1 = process_input_lines(Output_Grad4_Layer4_CH1)
    #print("Read Output_Grad4_Layer4_CH1")

    Output_Grad5_Layer4_CH0 = Read_DDR(Rd_Address=0x863C8000,  End_Address=0x863E2000)
    Output_Grad5_Layer4_CH0 = process_input_lines(Output_Grad5_Layer4_CH0)
    #print("Read Output_Grad5_Layer4_CH0")

    Output_Grad5_Layer4_CH1 = Read_DDR(Rd_Address=0x963C8000,  End_Address=0x963E2000)
    Output_Grad5_Layer4_CH1 = process_input_lines(Output_Grad5_Layer4_CH1)
    #print("Read Output_Grad5_Layer4_CH1")

    Output_Grad6_Layer4_CH0 = Read_DDR(Rd_Address=0x863E2000,  End_Address=0x863FC000)
    Output_Grad6_Layer4_CH0 = process_input_lines(Output_Grad6_Layer4_CH0)
    #print("Read Output_Grad6_Layer4_CH0")

    Output_Grad6_Layer4_CH1 = Read_DDR(Rd_Address=0x963E2000,  End_Address=0x963FC000)
    Output_Grad6_Layer4_CH1 = process_input_lines(Output_Grad6_Layer4_CH1)
    #print("Read Output_Grad6_Layer4_CH1")

    Output_Grad7_Layer4_CH0 = Read_DDR(Rd_Address=0x863FC000,  End_Address=0x86416000)
    Output_Grad7_Layer4_CH0 = process_input_lines(Output_Grad7_Layer4_CH0)
    #print("Read Output_Grad7_Layer4_CH0")

    Output_Grad7_Layer4_CH1 = Read_DDR(Rd_Address=0x963FC000,  End_Address=0x96416000)
    Output_Grad7_Layer4_CH1 = process_input_lines(Output_Grad7_Layer4_CH1)
    #print("Read Output_Grad7_Layer4_CH1")

    Output_Grad8_Layer4_CH0 = Read_DDR(Rd_Address=0x86416000,  End_Address=0x86430000)
    Output_Grad8_Layer4_CH0 = process_input_lines(Output_Grad8_Layer4_CH0)
    #print("Read Output_Grad8_Layer4_CH0")

    Output_Grad8_Layer4_CH1 = Read_DDR(Rd_Address=0x96416000,  End_Address=0x96430000)
    Output_Grad8_Layer4_CH1 = process_input_lines(Output_Grad8_Layer4_CH1)
    #print("Read Output_Grad8_Layer4_CH1")

    Output_Grad1_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer4_CH0, Output_Grad1_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad2_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer4_CH0, Output_Grad2_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad3_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer4_CH0, Output_Grad3_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad4_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer4_CH0, Output_Grad4_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad5_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer4_CH0, Output_Grad5_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad6_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer4_CH0, Output_Grad6_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad7_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer4_CH0, Output_Grad7_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grad8_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer4_CH0, Output_Grad8_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    Output_Grads_Layer4 = Output_Grad1_Layer4 + Output_Grad2_Layer4 + Output_Grad3_Layer4 + Output_Grad4_Layer4 + \
                            Output_Grad5_Layer4 + Output_Grad6_Layer4 + Output_Grad7_Layer4 + Output_Grad8_Layer4    
    Output_Grad_Layer4 = torch.tensor([float(value) for value in Output_Grads_Layer4], dtype=torch.float32).reshape(8, 128, 26, 26)

    # BReLu Marking
    ReLu_Marking1_Layer3_CH0 = Read_DDR(Rd_Address=0x87D2C000,  End_Address=0x87D46000)
    ReLu_Marking1_Layer3_CH0_256 = process_input_lines(ReLu_Marking1_Layer3_CH0)

    ReLu_Marking2_Layer3_CH0 = Read_DDR(Rd_Address=0x87D46000,  End_Address=0x87D60000)
    ReLu_Marking2_Layer3_CH0_256 = process_input_lines(ReLu_Marking2_Layer3_CH0)

    ReLu_Marking3_Layer3_CH0 = Read_DDR(Rd_Address=0x87D60000,  End_Address=0x87D7A000)
    ReLu_Marking3_Layer3_CH0_256 = process_input_lines(ReLu_Marking3_Layer3_CH0)

    ReLu_Marking4_Layer3_CH0 = Read_DDR(Rd_Address=0x87D7A000,  End_Address=0x87D94000)
    ReLu_Marking4_Layer3_CH0_256 = process_input_lines(ReLu_Marking4_Layer3_CH0)

    ReLu_Marking5_Layer3_CH0 = Read_DDR(Rd_Address=0x87D94000,  End_Address=0x87DAE000)
    ReLu_Marking5_Layer3_CH0_256 = process_input_lines(ReLu_Marking5_Layer3_CH0)

    ReLu_Marking6_Layer3_CH0 = Read_DDR(Rd_Address=0x87DAE000,  End_Address=0x87DC8000)
    ReLu_Marking6_Layer3_CH0_256 = process_input_lines(ReLu_Marking6_Layer3_CH0)

    ReLu_Marking7_Layer3_CH0 = Read_DDR(Rd_Address=0x87DC8000,  End_Address=0x87DE2000)
    ReLu_Marking7_Layer3_CH0_256 = process_input_lines(ReLu_Marking7_Layer3_CH0)

    ReLu_Marking8_Layer3_CH0 = Read_DDR(Rd_Address=0x87DE2000,  End_Address=0x87DFC000)
    ReLu_Marking8_Layer3_CH0_256 = process_input_lines(ReLu_Marking8_Layer3_CH0)

    ReLu_Marking1_Layer3_CH1 = Read_DDR(Rd_Address=0x97D2C000,  End_Address=0x97D46000)
    ReLu_Marking1_Layer3_CH1_256 = process_input_lines(ReLu_Marking1_Layer3_CH1)

    ReLu_Marking2_Layer3_CH1 = Read_DDR(Rd_Address=0x97D46000,  End_Address=0x97D60000)
    ReLu_Marking2_Layer3_CH1_256 = process_input_lines(ReLu_Marking2_Layer3_CH1)

    ReLu_Marking3_Layer3_CH1 = Read_DDR(Rd_Address=0x97D60000,  End_Address=0x97D7A000)
    ReLu_Marking3_Layer3_CH1_256 = process_input_lines(ReLu_Marking3_Layer3_CH1)

    ReLu_Marking4_Layer3_CH1 = Read_DDR(Rd_Address=0x97D7A000,  End_Address=0x97D94000)
    ReLu_Marking4_Layer3_CH1_256 = process_input_lines(ReLu_Marking4_Layer3_CH1)

    ReLu_Marking5_Layer3_CH1 = Read_DDR(Rd_Address=0x97D94000,  End_Address=0x97DAE000)
    ReLu_Marking5_Layer3_CH1_256 = process_input_lines(ReLu_Marking5_Layer3_CH1)

    ReLu_Marking6_Layer3_CH1 = Read_DDR(Rd_Address=0x97DAE000,  End_Address=0x97DC8000)
    ReLu_Marking6_Layer3_CH1_256 = process_input_lines(ReLu_Marking6_Layer3_CH1)

    ReLu_Marking7_Layer3_CH1 = Read_DDR(Rd_Address=0x97DC8000,  End_Address=0x97DE2000)
    ReLu_Marking7_Layer3_CH1_256 = process_input_lines(ReLu_Marking7_Layer3_CH1)

    ReLu_Marking8_Layer3_CH1 = Read_DDR(Rd_Address=0x97DE2000,  End_Address=0x97DFC000)
    ReLu_Marking8_Layer3_CH1_256 = process_input_lines(ReLu_Marking8_Layer3_CH1)

    # ReLu Reordering
    ReLu_Marking1_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer3_CH0_256, ReLu_Marking1_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking2_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer3_CH0_256, ReLu_Marking2_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking3_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer3_CH0_256, ReLu_Marking3_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking4_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer3_CH0_256, ReLu_Marking4_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking5_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer3_CH0_256, ReLu_Marking5_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking6_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer3_CH0_256, ReLu_Marking6_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking7_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer3_CH0_256, ReLu_Marking7_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    ReLu_Marking8_Layer3 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer3_CH0_256, ReLu_Marking8_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)


    ReLu_Marking_Layer3 = ReLu_Marking1_Layer3 + ReLu_Marking2_Layer3 + ReLu_Marking3_Layer3 + ReLu_Marking4_Layer3 + ReLu_Marking5_Layer3 + \
                            ReLu_Marking6_Layer3 + ReLu_Marking7_Layer3 + ReLu_Marking8_Layer3
    
    ReLu_Marking_Layer3 = torch.tensor([float(value) for value in ReLu_Marking_Layer3], dtype=torch.float32).reshape(8, 128, 26, 26)

    # BReLu Calculate
    # Output_Grad_layer4_input = torch.tensor(Output_Grad_Layer4, dtype=torch.float32).reshape(8,128,26,26)
    # Layer3_Location = torch.tensor(ReLu_Marking_Layer3, dtype=torch.float32).reshape(8,128,26,26)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer3)
    grad_relu_output = backward_active(Output_Grad_Layer4, relu_mask)
    grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_3, dL_dbeta_3, avg_pc_3, backward_const_3 = backward_LightNorm(grad_maxpool_output, layer3_cache)

    # avg_pc_3 = avg_pc_3.squeeze()
    # backward_const_3 = backward_const_3.squeeze()

    avg_pc_3, backward_const_3 = Mean_Var_Dec2Bfloat(avg_pc_3, backward_const_3, Exponent_Bits, Mantissa_Bits)

    # Weight_Backward_Layer3 for Soft2Hardware
    Weight_Backward_Layer3 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 64, Weight_List[3], backward_const_3, avg_pc_3)


    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer3_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer3[0])
    Weight_Backward_Layer3_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer3[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer3_CH0,Wr_Address=0x823D0000)
    Write_DDR(Weight_Backward_Layer3_CH1,Wr_Address=0x923D0000)


    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer4 = (Output_Grad_Layer4).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer4_CH0 = Read_DDR(Rd_Address=0x8F8DC000,  End_Address=0x8F924000)
    Weight_Gradient1_Layer4_CH0_256 = process_input_lines(Weight_Gradient1_Layer4_CH0)
    #print("Weight_Gradient1_Layer4_CH0 : ", len(Weight_Gradient1_Layer4_CH0))   

    Weight_Gradient2_Layer4_CH0 = Read_DDR(Rd_Address=0x8F924000,  End_Address=0x8F96C000)
    Weight_Gradient2_Layer4_CH0_256 = process_input_lines(Weight_Gradient2_Layer4_CH0)
    #print("Weight_Gradient2_Layer4_CH0 : ", len(Weight_Gradient2_Layer4_CH0))    

    Weight_Gradient3_Layer4_CH0 = Read_DDR(Rd_Address=0x8F96C000,  End_Address=0x8F9B4000)
    Weight_Gradient3_Layer4_CH0_256 = process_input_lines(Weight_Gradient3_Layer4_CH0)
    #print("Weight_Gradient3_Layer4_CH0 : ", len(Weight_Gradient3_Layer4_CH0)) 

    Weight_Gradient4_Layer4_CH0 = Read_DDR(Rd_Address=0x8F9B4000,  End_Address=0x8F9FC000)
    Weight_Gradient4_Layer4_CH0_256 = process_input_lines(Weight_Gradient4_Layer4_CH0)
    #print("Weight_Gradient4_Layer4_CH0 : ", len(Weight_Gradient4_Layer4_CH0)) 

    Weight_Gradient5_Layer4_CH0 = Read_DDR(Rd_Address=0x8F9FC000,  End_Address=0x8FA44000)
    Weight_Gradient5_Layer4_CH0_256 = process_input_lines(Weight_Gradient5_Layer4_CH0)
    #print("Weight_Gradient5_Layer4_CH0 : ", len(Weight_Gradient5_Layer4_CH0)) 

    Weight_Gradient6_Layer4_CH0 = Read_DDR(Rd_Address=0x8FA44000,  End_Address=0x8FA8C000)
    Weight_Gradient6_Layer4_CH0_256 = process_input_lines(Weight_Gradient6_Layer4_CH0)
    #print("Weight_Gradient6_Layer4_CH0 : ", len(Weight_Gradient6_Layer4_CH0)) 

    Weight_Gradient7_Layer4_CH0 = Read_DDR(Rd_Address=0x8FA8C000,  End_Address=0x8FAD4000)
    Weight_Gradient7_Layer4_CH0_256 = process_input_lines(Weight_Gradient7_Layer4_CH0)
    #print("Weight_Gradient7_Layer4_CH0 : ", len(Weight_Gradient7_Layer4_CH0)) 

    Weight_Gradient8_Layer4_CH0 = Read_DDR(Rd_Address=0x8FAD4000,  End_Address=0x8FB1C000)
    Weight_Gradient8_Layer4_CH0_256 = process_input_lines(Weight_Gradient8_Layer4_CH0)
    #print("Weight_Gradient8_Layer4_CH0 : ", len(Weight_Gradient8_Layer4_CH0)) 

    Weight_Gradient1_Layer4_CH1 = Read_DDR(Rd_Address=0x9F8DC000,  End_Address=0x9F924000)
    Weight_Gradient1_Layer4_CH1_256 = process_input_lines(Weight_Gradient1_Layer4_CH1)
    #print("Weight_Gradient1_Layer4_CH1 : ", len(Weight_Gradient1_Layer4_CH1)) 

    Weight_Gradient2_Layer4_CH1 = Read_DDR(Rd_Address=0x9F924000,  End_Address=0x9F96C000)
    Weight_Gradient2_Layer4_CH1_256 = process_input_lines(Weight_Gradient2_Layer4_CH1)
    #print("Weight_Gradient2_Layer4_CH1 : ", len(Weight_Gradient2_Layer4_CH1)) 

    Weight_Gradient3_Layer4_CH1 = Read_DDR(Rd_Address=0x9F96C000,  End_Address=0x9F9B4000)
    Weight_Gradient3_Layer4_CH1_256 = process_input_lines(Weight_Gradient3_Layer4_CH1)
    #print("Weight_Gradient3_Layer4_CH1 : ", len(Weight_Gradient3_Layer4_CH1)) 

    Weight_Gradient4_Layer4_CH1 = Read_DDR(Rd_Address=0x9F9B4000,  End_Address=0x9F9FC000)
    Weight_Gradient4_Layer4_CH1_256 = process_input_lines(Weight_Gradient4_Layer4_CH1)
    #print("Weight_Gradient4_Layer4_CH1 : ", len(Weight_Gradient4_Layer4_CH1)) 

    Weight_Gradient5_Layer4_CH1 = Read_DDR(Rd_Address=0x9F9FC000,  End_Address=0x9FA44000)
    Weight_Gradient5_Layer4_CH1_256 = process_input_lines(Weight_Gradient5_Layer4_CH1)
    #print("Weight_Gradient5_Layer4_CH1 : ", len(Weight_Gradient5_Layer4_CH1)) 

    Weight_Gradient6_Layer4_CH1 = Read_DDR(Rd_Address=0x9FA44000,  End_Address=0x9FA8C000)
    Weight_Gradient6_Layer4_CH1_256 = process_input_lines(Weight_Gradient6_Layer4_CH1)
    #print("Weight_Gradient6_Layer4_CH1 : ", len(Weight_Gradient6_Layer4_CH1)) 

    Weight_Gradient7_Layer4_CH1 = Read_DDR(Rd_Address=0x9FA8C000,  End_Address=0x9FAD4000)
    Weight_Gradient7_Layer4_CH1_256 = process_input_lines(Weight_Gradient7_Layer4_CH1)
    #print("Weight_Gradient7_Layer4_CH1 : ", len(Weight_Gradient7_Layer4_CH1)) 

    Weight_Gradient8_Layer4_CH1 = Read_DDR(Rd_Address=0x9FAD4000,  End_Address=0x9FB1C000)
    Weight_Gradient8_Layer4_CH1_256 = process_input_lines(Weight_Gradient8_Layer4_CH1)
    #print("Weight_Gradient8_Layer4_CH1 : ", len(Weight_Gradient8_Layer4_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer4_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer4_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer4_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer4_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

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
    Weight_Gradient_Layer4 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer4)]   
    Weight_Gradient_Layer4 = torch.tensor([float(value) for value in Weight_Gradient_Layer4], dtype=torch.float32).reshape(256, 128, 3, 3)   

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    #print(irq_val)

    #################################################
    #             Backward Layer 3 Start            #
    #################################################
    # check Layer3 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer3_CH0 = Read_DDR(Rd_Address=0x85B40000,  End_Address=0x85B74000)
    Output_Grad1_Layer3_CH0 = process_input_lines(Output_Grad1_Layer3_CH0)

    Output_Grad1_Layer3_CH1 = Read_DDR(Rd_Address=0x95B40000,  End_Address=0x95B74000)
    Output_Grad1_Layer3_CH1 = process_input_lines(Output_Grad1_Layer3_CH1)

    Output_Grad2_Layer3_CH0 = Read_DDR(Rd_Address=0x85B74000,  End_Address=0x85BA8000)
    Output_Grad2_Layer3_CH0 = process_input_lines(Output_Grad2_Layer3_CH0)

    Output_Grad2_Layer3_CH1 = Read_DDR(Rd_Address=0x95B74000,  End_Address=0x95BA8000)
    Output_Grad2_Layer3_CH1 = process_input_lines(Output_Grad2_Layer3_CH1)

    Output_Grad3_Layer3_CH0 = Read_DDR(Rd_Address=0x85BA8000,  End_Address=0x85BDC000)
    Output_Grad3_Layer3_CH0 = process_input_lines(Output_Grad3_Layer3_CH0)

    Output_Grad3_Layer3_CH1 = Read_DDR(Rd_Address=0x95BA8000,  End_Address=0x95BDC000)
    Output_Grad3_Layer3_CH1 = process_input_lines(Output_Grad3_Layer3_CH1)

    Output_Grad4_Layer3_CH0 = Read_DDR(Rd_Address=0x85BDC000,  End_Address=0x85C10000)
    Output_Grad4_Layer3_CH0 = process_input_lines(Output_Grad4_Layer3_CH0)

    Output_Grad4_Layer3_CH1 = Read_DDR(Rd_Address=0x95BDC000,  End_Address=0x95C10000)
    Output_Grad4_Layer3_CH1 = process_input_lines(Output_Grad4_Layer3_CH1)

    Output_Grad5_Layer3_CH0 = Read_DDR(Rd_Address=0x85C10000,  End_Address=0x85C44000)
    Output_Grad5_Layer3_CH0 = process_input_lines(Output_Grad5_Layer3_CH0)

    Output_Grad5_Layer3_CH1 = Read_DDR(Rd_Address=0x95C10000,  End_Address=0x95C44000)
    Output_Grad5_Layer3_CH1 = process_input_lines(Output_Grad5_Layer3_CH1)

    Output_Grad6_Layer3_CH0 = Read_DDR(Rd_Address=0x85C44000,  End_Address=0x85C78000)
    Output_Grad6_Layer3_CH0 = process_input_lines(Output_Grad6_Layer3_CH0)

    Output_Grad6_Layer3_CH1 = Read_DDR(Rd_Address=0x95C44000,  End_Address=0x95C78000)
    Output_Grad6_Layer3_CH1 = process_input_lines(Output_Grad6_Layer3_CH1)

    Output_Grad7_Layer3_CH0 = Read_DDR(Rd_Address=0x85C78000,  End_Address=0x85CAC000)
    Output_Grad7_Layer3_CH0 = process_input_lines(Output_Grad7_Layer3_CH0)

    Output_Grad7_Layer3_CH1 = Read_DDR(Rd_Address=0x95C78000,  End_Address=0x95CAC000)
    Output_Grad7_Layer3_CH1 = process_input_lines(Output_Grad7_Layer3_CH1)

    Output_Grad8_Layer3_CH0 = Read_DDR(Rd_Address=0x85CAC000,  End_Address=0x85CE0000)
    Output_Grad8_Layer3_CH0 = process_input_lines(Output_Grad8_Layer3_CH0)

    Output_Grad8_Layer3_CH1 = Read_DDR(Rd_Address=0x95CAC000,  End_Address=0x95CE0000)
    Output_Grad8_Layer3_CH1 = process_input_lines(Output_Grad8_Layer3_CH1)


    Output_Grad1_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer3_CH0, Output_Grad1_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad2_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer3_CH0, Output_Grad2_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad3_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer3_CH0, Output_Grad3_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad4_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer3_CH0, Output_Grad4_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad5_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer3_CH0, Output_Grad5_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad6_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer3_CH0, Output_Grad6_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad7_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer3_CH0, Output_Grad7_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grad8_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer3_CH0, Output_Grad8_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    Output_Grads_Layer3 = Output_Grad1_Layer3 + Output_Grad2_Layer3 + Output_Grad3_Layer3 + Output_Grad4_Layer3 + \
                            Output_Grad5_Layer3 + Output_Grad6_Layer3 + Output_Grad7_Layer3 + Output_Grad8_Layer3    
    Output_Grad_Layer3 = torch.tensor([float(value) for value in Output_Grads_Layer3], dtype=torch.float32).reshape(8, 64, 52, 52)

    # BReLu Marking
    ReLu_Marking1_Layer2_CH0 = Read_DDR(Rd_Address=0x87B8C000,  End_Address=0x87BC0000)
    ReLu_Marking1_Layer2_CH0_256 = process_input_lines(ReLu_Marking1_Layer2_CH0)

    ReLu_Marking2_Layer2_CH0 = Read_DDR(Rd_Address=0x87BC0000,  End_Address=0x87BF4000)
    ReLu_Marking2_Layer2_CH0_256 = process_input_lines(ReLu_Marking2_Layer2_CH0)

    ReLu_Marking3_Layer2_CH0 = Read_DDR(Rd_Address=0x87BF4000,  End_Address=0x87C28000)
    ReLu_Marking3_Layer2_CH0_256 = process_input_lines(ReLu_Marking3_Layer2_CH0)

    ReLu_Marking4_Layer2_CH0 = Read_DDR(Rd_Address=0x87C28000,  End_Address=0x87C5C000)
    ReLu_Marking4_Layer2_CH0_256 = process_input_lines(ReLu_Marking4_Layer2_CH0)

    ReLu_Marking5_Layer2_CH0 = Read_DDR(Rd_Address=0x87C5C000,  End_Address=0x87C90000)
    ReLu_Marking5_Layer2_CH0_256 = process_input_lines(ReLu_Marking5_Layer2_CH0)

    ReLu_Marking6_Layer2_CH0 = Read_DDR(Rd_Address=0x87C90000,  End_Address=0x87CC4000)
    ReLu_Marking6_Layer2_CH0_256 = process_input_lines(ReLu_Marking6_Layer2_CH0)

    ReLu_Marking7_Layer2_CH0 = Read_DDR(Rd_Address=0x87CC4000,  End_Address=0x87CF8000)
    ReLu_Marking7_Layer2_CH0_256 = process_input_lines(ReLu_Marking7_Layer2_CH0)

    ReLu_Marking8_Layer2_CH0 = Read_DDR(Rd_Address=0x87CF8000,  End_Address=0x87D2C000)
    ReLu_Marking8_Layer2_CH0_256 = process_input_lines(ReLu_Marking8_Layer2_CH0)

    ReLu_Marking1_Layer2_CH1 = Read_DDR(Rd_Address=0x97B8C000,  End_Address=0x97BC0000)
    ReLu_Marking1_Layer2_CH1_256 = process_input_lines(ReLu_Marking1_Layer2_CH1)

    ReLu_Marking2_Layer2_CH1 = Read_DDR(Rd_Address=0x97BC0000,  End_Address=0x97BF4000)
    ReLu_Marking2_Layer2_CH1_256 = process_input_lines(ReLu_Marking2_Layer2_CH1)

    ReLu_Marking3_Layer2_CH1 = Read_DDR(Rd_Address=0x97BF4000,  End_Address=0x97C28000)
    ReLu_Marking3_Layer2_CH1_256 = process_input_lines(ReLu_Marking3_Layer2_CH1)

    ReLu_Marking4_Layer2_CH1 = Read_DDR(Rd_Address=0x97C28000,  End_Address=0x97C5C000)
    ReLu_Marking4_Layer2_CH1_256 = process_input_lines(ReLu_Marking4_Layer2_CH1)

    ReLu_Marking5_Layer2_CH1 = Read_DDR(Rd_Address=0x97C5C000,  End_Address=0x97C90000)
    ReLu_Marking5_Layer2_CH1_256 = process_input_lines(ReLu_Marking5_Layer2_CH1)

    ReLu_Marking6_Layer2_CH1 = Read_DDR(Rd_Address=0x97C90000,  End_Address=0x97CC4000)
    ReLu_Marking6_Layer2_CH1_256 = process_input_lines(ReLu_Marking6_Layer2_CH1)

    ReLu_Marking7_Layer2_CH1 = Read_DDR(Rd_Address=0x97CC4000,  End_Address=0x97CF8000)
    ReLu_Marking7_Layer2_CH1_256 = process_input_lines(ReLu_Marking7_Layer2_CH1)

    ReLu_Marking8_Layer2_CH1 = Read_DDR(Rd_Address=0x97CF8000,  End_Address=0x97D2C000)
    ReLu_Marking8_Layer2_CH1_256 = process_input_lines(ReLu_Marking8_Layer2_CH1)

    # ReLu Reordering
    ReLu_Marking1_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer2_CH0_256, ReLu_Marking1_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking2_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer2_CH0_256, ReLu_Marking2_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking3_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer2_CH0_256, ReLu_Marking3_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking4_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer2_CH0_256, ReLu_Marking4_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking5_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer2_CH0_256, ReLu_Marking5_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking6_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer2_CH0_256, ReLu_Marking6_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking7_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer2_CH0_256, ReLu_Marking7_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    ReLu_Marking8_Layer2 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer2_CH0_256, ReLu_Marking8_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)


    ReLu_Marking_Layer2 = ReLu_Marking1_Layer2 + ReLu_Marking2_Layer2 + ReLu_Marking3_Layer2 + ReLu_Marking4_Layer2 + ReLu_Marking5_Layer2 + \
                            ReLu_Marking6_Layer2 + ReLu_Marking7_Layer2 + ReLu_Marking8_Layer2
    
    ReLu_Marking_Layer2 = torch.tensor([float(value) for value in ReLu_Marking_Layer2], dtype=torch.float32).reshape(8, 64, 52, 52)

    # BReLu Calculate
    # Output_Grad_layer3_input = torch.tensor(Output_Grad_Layer3, dtype=torch.float32).reshape(8,64,52,52)
    # Layer2_Location = torch.tensor(ReLu_Marking_Layer2, dtype=torch.float32).reshape(8,64,52,52)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer2)
    grad_relu_output = backward_active(Output_Grad_Layer3, relu_mask)
    grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_2, dL_dbeta_2, avg_pc_2, backward_const_2 = backward_LightNorm(grad_maxpool_output, layer2_cache)

    # avg_pc_2 = avg_pc_2.squeeze()
    # backward_const_2 = backward_const_2.squeeze()

    avg_pc_2, backward_const_2 = Mean_Var_Dec2Bfloat(avg_pc_2, backward_const_2, Exponent_Bits, Mantissa_Bits)


    # Weight_Backward_Layer2 for Soft2Hardware
    Weight_Backward_Layer2 = Weight_Hardware_Backward_ReOrdering_OtherLayer(64, 32, Weight_List[2], backward_const_2, avg_pc_2)


    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer2_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer2[0])
    Weight_Backward_Layer2_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer2[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer2_CH0,Wr_Address=0x823E4000)
    Write_DDR(Weight_Backward_Layer2_CH1,Wr_Address=0x923E4000)


    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer3 = (Output_Grad_Layer3).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB1C000,  End_Address=0x8FB2E000)
    Weight_Gradient1_Layer3_CH0_256 = process_input_lines(Weight_Gradient1_Layer3_CH0)
    #print("Weight_Gradient1_Layer3_CH0 : ", len(Weight_Gradient1_Layer3_CH0))   

    Weight_Gradient2_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB2E000,  End_Address=0x8FB40000)
    Weight_Gradient2_Layer3_CH0_256 = process_input_lines(Weight_Gradient2_Layer3_CH0)
    #print("Weight_Gradient2_Layer3_CH0 : ", len(Weight_Gradient2_Layer3_CH0))    

    Weight_Gradient3_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB40000,  End_Address=0x8FB52000)
    Weight_Gradient3_Layer3_CH0_256 = process_input_lines(Weight_Gradient3_Layer3_CH0)
    #print("Weight_Gradient3_Layer3_CH0 : ", len(Weight_Gradient3_Layer3_CH0)) 

    Weight_Gradient4_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB52000,  End_Address=0x8FB64000)
    Weight_Gradient4_Layer3_CH0_256 = process_input_lines(Weight_Gradient4_Layer3_CH0)
    #print("Weight_Gradient4_Layer3_CH0 : ", len(Weight_Gradient4_Layer3_CH0)) 

    Weight_Gradient5_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB64000,  End_Address=0x8FB76000)
    Weight_Gradient5_Layer3_CH0_256 = process_input_lines(Weight_Gradient5_Layer3_CH0)
    #print("Weight_Gradient5_Layer3_CH0 : ", len(Weight_Gradient5_Layer3_CH0)) 

    Weight_Gradient6_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB76000,  End_Address=0x8FB88000)
    Weight_Gradient6_Layer3_CH0_256 = process_input_lines(Weight_Gradient6_Layer3_CH0)
    #print("Weight_Gradient6_Layer3_CH0 : ", len(Weight_Gradient6_Layer3_CH0)) 

    Weight_Gradient7_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB88000,  End_Address=0x8FB9A000)
    Weight_Gradient7_Layer3_CH0_256 = process_input_lines(Weight_Gradient7_Layer3_CH0)
    #print("Weight_Gradient7_Layer3_CH0 : ", len(Weight_Gradient7_Layer3_CH0)) 

    Weight_Gradient8_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB9A000,  End_Address=0x8FBAC000)
    Weight_Gradient8_Layer3_CH0_256 = process_input_lines(Weight_Gradient8_Layer3_CH0)
    #print("Weight_Gradient8_Layer3_CH0 : ", len(Weight_Gradient8_Layer3_CH0)) 

    Weight_Gradient1_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB1C000,  End_Address=0x9FB2E000)
    Weight_Gradient1_Layer3_CH1_256 = process_input_lines(Weight_Gradient1_Layer3_CH1)
    #print("Weight_Gradient1_Layer3_CH1 : ", len(Weight_Gradient1_Layer3_CH1)) 

    Weight_Gradient2_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB2E000,  End_Address=0x9FB40000)
    Weight_Gradient2_Layer3_CH1_256 = process_input_lines(Weight_Gradient2_Layer3_CH1)
    #print("Weight_Gradient2_Layer3_CH1 : ", len(Weight_Gradient2_Layer3_CH1)) 

    Weight_Gradient3_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB40000,  End_Address=0x9FB52000)
    Weight_Gradient3_Layer3_CH1_256 = process_input_lines(Weight_Gradient3_Layer3_CH1)
    #print("Weight_Gradient3_Layer3_CH1 : ", len(Weight_Gradient3_Layer3_CH1)) 

    Weight_Gradient4_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB52000,  End_Address=0x9FB64000)
    Weight_Gradient4_Layer3_CH1_256 = process_input_lines(Weight_Gradient4_Layer3_CH1)
    #print("Weight_Gradient4_Layer3_CH1 : ", len(Weight_Gradient4_Layer3_CH1)) 

    Weight_Gradient5_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB64000,  End_Address=0x9FB76000)
    Weight_Gradient5_Layer3_CH1_256 = process_input_lines(Weight_Gradient5_Layer3_CH1)
    #print("Weight_Gradient5_Layer3_CH1 : ", len(Weight_Gradient5_Layer3_CH1)) 

    Weight_Gradient6_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB76000,  End_Address=0x9FB88000)
    Weight_Gradient6_Layer3_CH1_256 = process_input_lines(Weight_Gradient6_Layer3_CH1)
    #print("Weight_Gradient6_Layer3_CH1 : ", len(Weight_Gradient6_Layer3_CH1)) 

    Weight_Gradient7_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB88000,  End_Address=0x9FB9A000)
    Weight_Gradient7_Layer3_CH1_256 = process_input_lines(Weight_Gradient7_Layer3_CH1)
    #print("Weight_Gradient7_Layer3_CH1 : ", len(Weight_Gradient7_Layer3_CH1)) 

    Weight_Gradient8_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB9A000,  End_Address=0x9FBAC000)
    Weight_Gradient8_Layer3_CH1_256 = process_input_lines(Weight_Gradient8_Layer3_CH1)
    #print("Weight_Gradient8_Layer3_CH1 : ", len(Weight_Gradient8_Layer3_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer3_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer3_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer3_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer3_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

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
    Weight_Gradient_Layer3 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer3)]   
    Weight_Gradient_Layer3 = torch.tensor([float(value) for value in Weight_Gradient_Layer3], dtype=torch.float32).reshape(128, 64, 3, 3)   

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    #print(irq_val)

    #################################################
    #             Backward Layer 2 Start            #
    #################################################
    # check Layer2 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer2_CH0 = Read_DDR(Rd_Address=0x84B00000,  End_Address=0x84B68000)
    Output_Grad1_Layer2_CH0 = process_input_lines(Output_Grad1_Layer2_CH0)

    Output_Grad1_Layer2_CH1 = Read_DDR(Rd_Address=0x94B00000,  End_Address=0x94B68000)
    Output_Grad1_Layer2_CH1 = process_input_lines(Output_Grad1_Layer2_CH1)

    Output_Grad2_Layer2_CH0 = Read_DDR(Rd_Address=0x84B68000,  End_Address=0x84BD0000)
    Output_Grad2_Layer2_CH0 = process_input_lines(Output_Grad2_Layer2_CH0)

    Output_Grad2_Layer2_CH1 = Read_DDR(Rd_Address=0x94B68000,  End_Address=0x94BD0000)
    Output_Grad2_Layer2_CH1 = process_input_lines(Output_Grad2_Layer2_CH1)

    Output_Grad3_Layer2_CH0 = Read_DDR(Rd_Address=0x84BD0000,  End_Address=0x84C38000)
    Output_Grad3_Layer2_CH0 = process_input_lines(Output_Grad3_Layer2_CH0)

    Output_Grad3_Layer2_CH1 = Read_DDR(Rd_Address=0x94BD0000,  End_Address=0x94C38000)
    Output_Grad3_Layer2_CH1 = process_input_lines(Output_Grad3_Layer2_CH1)

    Output_Grad4_Layer2_CH0 = Read_DDR(Rd_Address=0x84C38000,  End_Address=0x84CA0000)
    Output_Grad4_Layer2_CH0 = process_input_lines(Output_Grad4_Layer2_CH0)

    Output_Grad4_Layer2_CH1 = Read_DDR(Rd_Address=0x94C38000,  End_Address=0x94CA0000)
    Output_Grad4_Layer2_CH1 = process_input_lines(Output_Grad4_Layer2_CH1)

    Output_Grad5_Layer2_CH0 = Read_DDR(Rd_Address=0x84CA0000,  End_Address=0x84D08000)
    Output_Grad5_Layer2_CH0 = process_input_lines(Output_Grad5_Layer2_CH0)

    Output_Grad5_Layer2_CH1 = Read_DDR(Rd_Address=0x94CA0000,  End_Address=0x94D08000)
    Output_Grad5_Layer2_CH1 = process_input_lines(Output_Grad5_Layer2_CH1)

    Output_Grad6_Layer2_CH0 = Read_DDR(Rd_Address=0x84D08000,  End_Address=0x84D70000)
    Output_Grad6_Layer2_CH0 = process_input_lines(Output_Grad6_Layer2_CH0)

    Output_Grad6_Layer2_CH1 = Read_DDR(Rd_Address=0x94D08000,  End_Address=0x94D70000)
    Output_Grad6_Layer2_CH1 = process_input_lines(Output_Grad6_Layer2_CH1)

    Output_Grad7_Layer2_CH0 = Read_DDR(Rd_Address=0x84D70000,  End_Address=0x84DD8000)
    Output_Grad7_Layer2_CH0 = process_input_lines(Output_Grad7_Layer2_CH0)

    Output_Grad7_Layer2_CH1 = Read_DDR(Rd_Address=0x94D70000,  End_Address=0x94DD8000)
    Output_Grad7_Layer2_CH1 = process_input_lines(Output_Grad7_Layer2_CH1)

    Output_Grad8_Layer2_CH0 = Read_DDR(Rd_Address=0x84DD8000,  End_Address=0x84E40000)
    Output_Grad8_Layer2_CH0 = process_input_lines(Output_Grad8_Layer2_CH0)

    Output_Grad8_Layer2_CH1 = Read_DDR(Rd_Address=0x94DD8000,  End_Address=0x94E40000)
    Output_Grad8_Layer2_CH1 = process_input_lines(Output_Grad8_Layer2_CH1)


    Output_Grad1_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer2_CH0, Output_Grad1_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad2_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer2_CH0, Output_Grad2_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad3_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer2_CH0, Output_Grad3_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad4_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer2_CH0, Output_Grad4_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad5_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer2_CH0, Output_Grad5_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad6_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer2_CH0, Output_Grad6_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad7_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer2_CH0, Output_Grad7_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grad8_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer2_CH0, Output_Grad8_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    Output_Grads_Layer2 = Output_Grad1_Layer2 + Output_Grad2_Layer2 + Output_Grad3_Layer2 + Output_Grad4_Layer2 + \
                            Output_Grad5_Layer2 + Output_Grad6_Layer2 + Output_Grad7_Layer2 + Output_Grad8_Layer2    
    Output_Grad_Layer2 = torch.tensor([float(value) for value in Output_Grads_Layer2], dtype=torch.float32).reshape(8, 32, 104, 104)

    # BReLu Marking
    ReLu_Marking1_Layer1_CH0 = Read_DDR(Rd_Address=0x8784C000,  End_Address=0x878B4000)
    ReLu_Marking1_Layer1_CH0_256 = process_input_lines(ReLu_Marking1_Layer1_CH0)

    ReLu_Marking2_Layer1_CH0 = Read_DDR(Rd_Address=0x878B4000,  End_Address=0x8791C000)
    ReLu_Marking2_Layer1_CH0_256 = process_input_lines(ReLu_Marking2_Layer1_CH0)

    ReLu_Marking3_Layer1_CH0 = Read_DDR(Rd_Address=0x8791C000,  End_Address=0x87984000)
    ReLu_Marking3_Layer1_CH0_256 = process_input_lines(ReLu_Marking3_Layer1_CH0)

    ReLu_Marking4_Layer1_CH0 = Read_DDR(Rd_Address=0x87984000,  End_Address=0x879EC000)
    ReLu_Marking4_Layer1_CH0_256 = process_input_lines(ReLu_Marking4_Layer1_CH0)

    ReLu_Marking5_Layer1_CH0 = Read_DDR(Rd_Address=0x879EC000,  End_Address=0x87A54000)
    ReLu_Marking5_Layer1_CH0_256 = process_input_lines(ReLu_Marking5_Layer1_CH0)

    ReLu_Marking6_Layer1_CH0 = Read_DDR(Rd_Address=0x87A54000,  End_Address=0x87ABC000)
    ReLu_Marking6_Layer1_CH0_256 = process_input_lines(ReLu_Marking6_Layer1_CH0)

    ReLu_Marking7_Layer1_CH0 = Read_DDR(Rd_Address=0x87ABC000,  End_Address=0x87B24000)
    ReLu_Marking7_Layer1_CH0_256 = process_input_lines(ReLu_Marking7_Layer1_CH0)

    ReLu_Marking8_Layer1_CH0 = Read_DDR(Rd_Address=0x87B24000,  End_Address=0x87B8C000)
    ReLu_Marking8_Layer1_CH0_256 = process_input_lines(ReLu_Marking8_Layer1_CH0)

    ReLu_Marking1_Layer1_CH1 = Read_DDR(Rd_Address=0x9784C000,  End_Address=0x978B4000)
    ReLu_Marking1_Layer1_CH1_256 = process_input_lines(ReLu_Marking1_Layer1_CH1)

    ReLu_Marking2_Layer1_CH1 = Read_DDR(Rd_Address=0x978B4000,  End_Address=0x9791C000)
    ReLu_Marking2_Layer1_CH1_256 = process_input_lines(ReLu_Marking2_Layer1_CH1)

    ReLu_Marking3_Layer1_CH1 = Read_DDR(Rd_Address=0x9791C000,  End_Address=0x97984000)
    ReLu_Marking3_Layer1_CH1_256 = process_input_lines(ReLu_Marking3_Layer1_CH1)

    ReLu_Marking4_Layer1_CH1 = Read_DDR(Rd_Address=0x97984000,  End_Address=0x979EC000)
    ReLu_Marking4_Layer1_CH1_256 = process_input_lines(ReLu_Marking4_Layer1_CH1)

    ReLu_Marking5_Layer1_CH1 = Read_DDR(Rd_Address=0x979EC000,  End_Address=0x97A54000)
    ReLu_Marking5_Layer1_CH1_256 = process_input_lines(ReLu_Marking5_Layer1_CH1)

    ReLu_Marking6_Layer1_CH1 = Read_DDR(Rd_Address=0x97A54000,  End_Address=0x97ABC000)
    ReLu_Marking6_Layer1_CH1_256 = process_input_lines(ReLu_Marking6_Layer1_CH1)

    ReLu_Marking7_Layer1_CH1 = Read_DDR(Rd_Address=0x97ABC000,  End_Address=0x97B24000)
    ReLu_Marking7_Layer1_CH1_256 = process_input_lines(ReLu_Marking7_Layer1_CH1)

    ReLu_Marking8_Layer1_CH1 = Read_DDR(Rd_Address=0x97B24000,  End_Address=0x97B8C000)
    ReLu_Marking8_Layer1_CH1_256 = process_input_lines(ReLu_Marking8_Layer1_CH1)

    # ReLu Reordering
    ReLu_Marking1_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer1_CH0_256, ReLu_Marking1_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking2_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer1_CH0_256, ReLu_Marking2_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking3_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer1_CH0_256, ReLu_Marking3_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking4_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer1_CH0_256, ReLu_Marking4_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking5_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer1_CH0_256, ReLu_Marking5_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking6_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer1_CH0_256, ReLu_Marking6_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking7_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer1_CH0_256, ReLu_Marking7_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    ReLu_Marking8_Layer1 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer1_CH0_256, ReLu_Marking8_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)


    ReLu_Marking_Layer1 = ReLu_Marking1_Layer1 + ReLu_Marking2_Layer1 + ReLu_Marking3_Layer1 + ReLu_Marking4_Layer1 + ReLu_Marking5_Layer1 + \
                            ReLu_Marking6_Layer1 + ReLu_Marking7_Layer1 + ReLu_Marking8_Layer1
    
    ReLu_Marking_Layer1 = torch.tensor([float(value) for value in ReLu_Marking_Layer1], dtype=torch.float32).reshape(8, 32, 104, 104)

    # BReLu Calculate
    # Output_Grad_Layer2_input = torch.tensor(Output_Grad_Layer2, dtype=torch.float32).reshape(8,32,104,104)
    # Layer1_Location = torch.tensor(ReLu_Marking_Layer1, dtype=torch.float32).reshape(8,32,104,104)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer1)
    grad_relu_output = backward_active(Output_Grad_Layer2, relu_mask)
    grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_1, dL_dbeta_1, avg_pc_1, backward_const_1 = backward_LightNorm(grad_maxpool_output, layer1_cache)

    # avg_pc_1 = avg_pc_1.squeeze()
    # backward_const_1 = backward_const_1.squeeze()

    avg_pc_1, backward_const_1 = Mean_Var_Dec2Bfloat(avg_pc_1, backward_const_1, Exponent_Bits, Mantissa_Bits)


    # Weight_Backward_Layer1 for Soft2Hardware
    Weight_Backward_Layer1 = Weight_Hardware_Backward_ReOrdering_OtherLayer(32, 16, Weight_List[1], backward_const_1, avg_pc_1)


    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer1_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer1[0])
    Weight_Backward_Layer1_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer1[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer1_CH0,Wr_Address=0x823E9000)
    Write_DDR(Weight_Backward_Layer1_CH1,Wr_Address=0x923E9000)


    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer2 = (Output_Grad_Layer2).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBAC000,  End_Address=0x8FBB0800)
    Weight_Gradient1_Layer2_CH0_256 = process_input_lines(Weight_Gradient1_Layer2_CH0)
    #print("Weight_Gradient1_Layer2_CH0 : ", len(Weight_Gradient1_Layer2_CH0))   

    Weight_Gradient2_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBB0800,  End_Address=0x8FBB5000)
    Weight_Gradient2_Layer2_CH0_256 = process_input_lines(Weight_Gradient2_Layer2_CH0)
    #print("Weight_Gradient2_Layer2_CH0 : ", len(Weight_Gradient2_Layer2_CH0))    

    Weight_Gradient3_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBB5000,  End_Address=0x8FBB9800)
    Weight_Gradient3_Layer2_CH0_256 = process_input_lines(Weight_Gradient3_Layer2_CH0)
    #print("Weight_Gradient3_Layer2_CH0 : ", len(Weight_Gradient3_Layer2_CH0)) 

    Weight_Gradient4_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBB9800,  End_Address=0x8FBBE000)
    Weight_Gradient4_Layer2_CH0_256 = process_input_lines(Weight_Gradient4_Layer2_CH0)
    #print("Weight_Gradient4_Layer2_CH0 : ", len(Weight_Gradient4_Layer2_CH0)) 

    Weight_Gradient5_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBBE000,  End_Address=0x8FBC2800)
    Weight_Gradient5_Layer2_CH0_256 = process_input_lines(Weight_Gradient5_Layer2_CH0)
    #print("Weight_Gradient5_Layer2_CH0 : ", len(Weight_Gradient5_Layer2_CH0)) 

    Weight_Gradient6_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBC2800,  End_Address=0x8FBC7000)
    Weight_Gradient6_Layer2_CH0_256 = process_input_lines(Weight_Gradient6_Layer2_CH0)
    #print("Weight_Gradient6_Layer2_CH0 : ", len(Weight_Gradient6_Layer2_CH0)) 

    Weight_Gradient7_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBC7000,  End_Address=0x8FBCB800)
    Weight_Gradient7_Layer2_CH0_256 = process_input_lines(Weight_Gradient7_Layer2_CH0)
    #print("Weight_Gradient7_Layer2_CH0 : ", len(Weight_Gradient7_Layer2_CH0)) 

    Weight_Gradient8_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBCB800,  End_Address=0x8FBD0000)
    Weight_Gradient8_Layer2_CH0_256 = process_input_lines(Weight_Gradient8_Layer2_CH0)
    #print("Weight_Gradient8_Layer2_CH0 : ", len(Weight_Gradient8_Layer2_CH0)) 

    Weight_Gradient1_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBAC000,  End_Address=0x9FBB0800)
    Weight_Gradient1_Layer2_CH1_256 = process_input_lines(Weight_Gradient1_Layer2_CH1)
    #print("Weight_Gradient1_Layer2_CH1 : ", len(Weight_Gradient1_Layer2_CH1)) 

    Weight_Gradient2_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBB0800,  End_Address=0x9FBB5000)
    Weight_Gradient2_Layer2_CH1_256 = process_input_lines(Weight_Gradient2_Layer2_CH1)
    #print("Weight_Gradient2_Layer2_CH1 : ", len(Weight_Gradient2_Layer2_CH1)) 

    Weight_Gradient3_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBB5000,  End_Address=0x9FBB9800)
    Weight_Gradient3_Layer2_CH1_256 = process_input_lines(Weight_Gradient3_Layer2_CH1)
    #print("Weight_Gradient3_Layer2_CH1 : ", len(Weight_Gradient3_Layer2_CH1)) 

    Weight_Gradient4_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBB9800,  End_Address=0x9FBBE000)
    Weight_Gradient4_Layer2_CH1_256 = process_input_lines(Weight_Gradient4_Layer2_CH1)
    #print("Weight_Gradient4_Layer2_CH1 : ", len(Weight_Gradient4_Layer2_CH1)) 

    Weight_Gradient5_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBBE000,  End_Address=0x9FBC2800)
    Weight_Gradient5_Layer2_CH1_256 = process_input_lines(Weight_Gradient5_Layer2_CH1)
    #print("Weight_Gradient5_Layer2_CH1 : ", len(Weight_Gradient5_Layer2_CH1)) 

    Weight_Gradient6_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBC2800,  End_Address=0x9FBC7000)
    Weight_Gradient6_Layer2_CH1_256 = process_input_lines(Weight_Gradient6_Layer2_CH1)
    #print("Weight_Gradient6_Layer2_CH1 : ", len(Weight_Gradient6_Layer2_CH1)) 

    Weight_Gradient7_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBC7000,  End_Address=0x9FBCB800)
    Weight_Gradient7_Layer2_CH1_256 = process_input_lines(Weight_Gradient7_Layer2_CH1)
    #print("Weight_Gradient7_Layer2_CH1 : ", len(Weight_Gradient7_Layer2_CH1)) 

    Weight_Gradient8_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBCB800,  End_Address=0x9FBD0000)
    Weight_Gradient8_Layer2_CH1_256 = process_input_lines(Weight_Gradient8_Layer2_CH1)
    #print("Weight_Gradient8_Layer2_CH1 : ", len(Weight_Gradient8_Layer2_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer2_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer2_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer2_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer2_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

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
    Weight_Gradient_Layer2 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer2)]   
    Weight_Gradient_Layer2 = torch.tensor([float(value) for value in Weight_Gradient_Layer2], dtype=torch.float32).reshape(64, 32, 3, 3)   

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()
    #print(irq_val)

    #################################################
    #             Backward Layer 1 Start            #
    #################################################
    # check Layer1 IRQ
    check_irq_otherlayer()

    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer1_CH0 = Read_DDR(Rd_Address=0x83E00000,  End_Address=0x83ED0000)
    Output_Grad1_Layer1_CH0 = process_input_lines(Output_Grad1_Layer1_CH0)

    Output_Grad1_Layer1_CH1 = Read_DDR(Rd_Address=0x93E00000,  End_Address=0x93ED0000)
    Output_Grad1_Layer1_CH1 = process_input_lines(Output_Grad1_Layer1_CH1)

    Output_Grad2_Layer1_CH0 = Read_DDR(Rd_Address=0x83ED0000,  End_Address=0x83FA0000)
    Output_Grad2_Layer1_CH0 = process_input_lines(Output_Grad2_Layer1_CH0)

    Output_Grad2_Layer1_CH1 = Read_DDR(Rd_Address=0x93ED0000,  End_Address=0x93FA0000)
    Output_Grad2_Layer1_CH1 = process_input_lines(Output_Grad2_Layer1_CH1)

    Output_Grad3_Layer1_CH0 = Read_DDR(Rd_Address=0x83FA0000,  End_Address=0x84070000)
    Output_Grad3_Layer1_CH0 = process_input_lines(Output_Grad3_Layer1_CH0)

    Output_Grad3_Layer1_CH1 = Read_DDR(Rd_Address=0x93FA0000,  End_Address=0x94070000)
    Output_Grad3_Layer1_CH1 = process_input_lines(Output_Grad3_Layer1_CH1)

    Output_Grad4_Layer1_CH0 = Read_DDR(Rd_Address=0x84070000,  End_Address=0x84140000)
    Output_Grad4_Layer1_CH0 = process_input_lines(Output_Grad4_Layer1_CH0)

    Output_Grad4_Layer1_CH1 = Read_DDR(Rd_Address=0x94070000,  End_Address=0x94140000)
    Output_Grad4_Layer1_CH1 = process_input_lines(Output_Grad4_Layer1_CH1)

    Output_Grad5_Layer1_CH0 = Read_DDR(Rd_Address=0x84140000,  End_Address=0x84210000)
    Output_Grad5_Layer1_CH0 = process_input_lines(Output_Grad5_Layer1_CH0)

    Output_Grad5_Layer1_CH1 = Read_DDR(Rd_Address=0x94140000,  End_Address=0x94210000)
    Output_Grad5_Layer1_CH1 = process_input_lines(Output_Grad5_Layer1_CH1)

    Output_Grad6_Layer1_CH0 = Read_DDR(Rd_Address=0x84210000,  End_Address=0x842E0000)
    Output_Grad6_Layer1_CH0 = process_input_lines(Output_Grad6_Layer1_CH0)

    Output_Grad6_Layer1_CH1 = Read_DDR(Rd_Address=0x94210000,  End_Address=0x942E0000)
    Output_Grad6_Layer1_CH1 = process_input_lines(Output_Grad6_Layer1_CH1)

    Output_Grad7_Layer1_CH0 = Read_DDR(Rd_Address=0x842E0000,  End_Address=0x843B0000)
    Output_Grad7_Layer1_CH0 = process_input_lines(Output_Grad7_Layer1_CH0)

    Output_Grad7_Layer1_CH1 = Read_DDR(Rd_Address=0x942E0000,  End_Address=0x943B0000)
    Output_Grad7_Layer1_CH1 = process_input_lines(Output_Grad7_Layer1_CH1)

    Output_Grad8_Layer1_CH0 = Read_DDR(Rd_Address=0x843B0000,  End_Address=0x84480000)
    Output_Grad8_Layer1_CH0 = process_input_lines(Output_Grad8_Layer1_CH0)

    Output_Grad8_Layer1_CH1 = Read_DDR(Rd_Address=0x943B0000,  End_Address=0x94480000)
    Output_Grad8_Layer1_CH1 = process_input_lines(Output_Grad8_Layer1_CH1)

    print("Convert Output Gradient")
    Output_Grad1_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer1_CH0, Output_Grad1_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad2_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer1_CH0, Output_Grad2_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad3_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer1_CH0, Output_Grad3_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad4_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer1_CH0, Output_Grad4_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad5_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer1_CH0, Output_Grad5_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad6_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer1_CH0, Output_Grad6_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad7_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer1_CH0, Output_Grad7_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grad8_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer1_CH0, Output_Grad8_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    Output_Grads_Layer1 = Output_Grad1_Layer1 + Output_Grad2_Layer1 + Output_Grad3_Layer1 + Output_Grad4_Layer1 + \
                            Output_Grad5_Layer1 + Output_Grad6_Layer1 + Output_Grad7_Layer1 + Output_Grad8_Layer1    
    Output_Grad_Layer1 = torch.tensor([float(value) for value in Output_Grads_Layer1], dtype=torch.float32).reshape(8, 16, 208, 208)

    # BReLu Marking
    ReLu_Marking1_Layer0_CH0 = Read_DDR(Rd_Address=0x871CC000,  End_Address=0x8729C000)
    ReLu_Marking1_Layer0_CH0_256 = process_input_lines(ReLu_Marking1_Layer0_CH0)

    ReLu_Marking2_Layer0_CH0 = Read_DDR(Rd_Address=0x8729C000,  End_Address=0x8736C000)
    ReLu_Marking2_Layer0_CH0_256 = process_input_lines(ReLu_Marking2_Layer0_CH0)

    ReLu_Marking3_Layer0_CH0 = Read_DDR(Rd_Address=0x8736C000,  End_Address=0x8743C000)
    ReLu_Marking3_Layer0_CH0_256 = process_input_lines(ReLu_Marking3_Layer0_CH0)

    ReLu_Marking4_Layer0_CH0 = Read_DDR(Rd_Address=0x8743C000,  End_Address=0x8750C000)
    ReLu_Marking4_Layer0_CH0_256 = process_input_lines(ReLu_Marking4_Layer0_CH0)

    ReLu_Marking5_Layer0_CH0 = Read_DDR(Rd_Address=0x8750C000,  End_Address=0x875DC000)
    ReLu_Marking5_Layer0_CH0_256 = process_input_lines(ReLu_Marking5_Layer0_CH0)

    ReLu_Marking6_Layer0_CH0 = Read_DDR(Rd_Address=0x875DC000,  End_Address=0x876AC000)
    ReLu_Marking6_Layer0_CH0_256 = process_input_lines(ReLu_Marking6_Layer0_CH0)

    ReLu_Marking7_Layer0_CH0 = Read_DDR(Rd_Address=0x876AC000,  End_Address=0x8777C000)
    ReLu_Marking7_Layer0_CH0_256 = process_input_lines(ReLu_Marking7_Layer0_CH0)

    ReLu_Marking8_Layer0_CH0 = Read_DDR(Rd_Address=0x8777C000,  End_Address=0x8784C000)
    ReLu_Marking8_Layer0_CH0_256 = process_input_lines(ReLu_Marking8_Layer0_CH0)

    ReLu_Marking1_Layer0_CH1 = Read_DDR(Rd_Address=0x971CC000,  End_Address=0x9729C000)
    ReLu_Marking1_Layer0_CH1_256 = process_input_lines(ReLu_Marking1_Layer0_CH1)

    ReLu_Marking2_Layer0_CH1 = Read_DDR(Rd_Address=0x9729C000,  End_Address=0x9736C000)
    ReLu_Marking2_Layer0_CH1_256 = process_input_lines(ReLu_Marking2_Layer0_CH1)

    ReLu_Marking3_Layer0_CH1 = Read_DDR(Rd_Address=0x9736C000,  End_Address=0x9743C000)
    ReLu_Marking3_Layer0_CH1_256 = process_input_lines(ReLu_Marking3_Layer0_CH1)

    ReLu_Marking4_Layer0_CH1 = Read_DDR(Rd_Address=0x9743C000,  End_Address=0x9750C000)
    ReLu_Marking4_Layer0_CH1_256 = process_input_lines(ReLu_Marking4_Layer0_CH1)

    ReLu_Marking5_Layer0_CH1 = Read_DDR(Rd_Address=0x9750C000,  End_Address=0x975DC000)
    ReLu_Marking5_Layer0_CH1_256 = process_input_lines(ReLu_Marking5_Layer0_CH1)

    ReLu_Marking6_Layer0_CH1 = Read_DDR(Rd_Address=0x975DC000,  End_Address=0x976AC000)
    ReLu_Marking6_Layer0_CH1_256 = process_input_lines(ReLu_Marking6_Layer0_CH1)

    ReLu_Marking7_Layer0_CH1 = Read_DDR(Rd_Address=0x976AC000,  End_Address=0x9777C000)
    ReLu_Marking7_Layer0_CH1_256 = process_input_lines(ReLu_Marking7_Layer0_CH1)

    ReLu_Marking8_Layer0_CH1 = Read_DDR(Rd_Address=0x9777C000,  End_Address=0x9784C000)
    ReLu_Marking8_Layer0_CH1_256 = process_input_lines(ReLu_Marking8_Layer0_CH1)

    # ReLu Reordering
    print("Convert ReLu masking")
    ReLu_Marking1_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking1_Layer0_CH0_256, ReLu_Marking1_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking2_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking2_Layer0_CH0_256, ReLu_Marking2_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking3_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking3_Layer0_CH0_256, ReLu_Marking3_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking4_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking4_Layer0_CH0_256, ReLu_Marking4_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking5_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking5_Layer0_CH0_256, ReLu_Marking5_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking6_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking6_Layer0_CH0_256, ReLu_Marking6_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking7_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking7_Layer0_CH0_256, ReLu_Marking7_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    ReLu_Marking8_Layer0 = Read_OutFmap_Bfloat2Dec(ReLu_Marking8_Layer0_CH0_256, ReLu_Marking8_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)


    ReLu_Marking_Layer0 = ReLu_Marking1_Layer0 + ReLu_Marking2_Layer0 + ReLu_Marking3_Layer0 + ReLu_Marking4_Layer0 + ReLu_Marking5_Layer0 + \
                            ReLu_Marking6_Layer0 + ReLu_Marking7_Layer0 + ReLu_Marking8_Layer0
    
    ReLu_Marking_Layer0 = torch.tensor([float(value) for value in ReLu_Marking_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)

    # BReLu Calculate
    # Output_Grad_Layer1_input = torch.tensor(Output_Grad_Layer1, dtype=torch.float32).reshape(8,16,208,208)
    # Layer0_Location = torch.tensor(ReLu_Marking_Layer0, dtype=torch.float32).reshape(8,16,208,208)

    relu_mask, location_mask = split_location(ReLu_Marking_Layer0)
    grad_relu_output = backward_active(Output_Grad_Layer1, relu_mask)
    grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
    dL_dgamma_0, dL_dbeta_0, avg_pc_0, backward_const_0 = backward_LightNorm(grad_relu_output, layer0_cache)

    # avg_pc_0 = avg_pc_0.squeeze()
    # backward_const_0 = backward_const_0.squeeze()

    avg_pc_0, backward_const_0 = Mean_Var_Dec2Bfloat(avg_pc_0, backward_const_0, Exponent_Bits, Mantissa_Bits)


    # Weight_Backward_Layer0 for Soft2Hardware
    Weight_Backward_Layer0 = Weight_Hardware_Backward_ReOrdering_Layer0(16, 16, Weight_List[0], backward_const_0, avg_pc_0)
    #print("Weight_Backward_Layer0: " + str(len(Weight_Backward_Layer0[0])))
    #print("Weight_Backward_Layer0: " + str(len(Weight_Backward_Layer0[1])))

    # Break 256To32 and Flip the Data: 
    Weight_Backward_Layer0_CH0 = Break_FlipHex_256To32(Weight_Backward_Layer0[0])
    Weight_Backward_Layer0_CH1 = Break_FlipHex_256To32(Weight_Backward_Layer0[1])

    # Write Weight For Backward into DDR
    Write_DDR(Weight_Backward_Layer0_CH0,Wr_Address=0x823EA400)
    Write_DDR(Weight_Backward_Layer0_CH1,Wr_Address=0x923EA400)


    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer1 = (Output_Grad_Layer1).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD0000,  End_Address=0x8FBD1200)
    Weight_Gradient1_Layer1_CH0_256 = process_input_lines(Weight_Gradient1_Layer1_CH0)
    #print("Weight_Gradient1_Layer1_CH0 : ", len(Weight_Gradient1_Layer1_CH0))   

    Weight_Gradient2_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD1200,  End_Address=0x8FBD2400)
    Weight_Gradient2_Layer1_CH0_256 = process_input_lines(Weight_Gradient2_Layer1_CH0)
    #print("Weight_Gradient2_Layer1_CH0 : ", len(Weight_Gradient2_Layer1_CH0))    

    Weight_Gradient3_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD2400,  End_Address=0x8FBD3600)
    Weight_Gradient3_Layer1_CH0_256 = process_input_lines(Weight_Gradient3_Layer1_CH0)
    #print("Weight_Gradient3_Layer1_CH0 : ", len(Weight_Gradient3_Layer1_CH0)) 

    Weight_Gradient4_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD3600,  End_Address=0x8FBD4800)
    Weight_Gradient4_Layer1_CH0_256 = process_input_lines(Weight_Gradient4_Layer1_CH0)
    #print("Weight_Gradient4_Layer1_CH0 : ", len(Weight_Gradient4_Layer1_CH0)) 

    Weight_Gradient5_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD4800,  End_Address=0x8FBD5A00)
    Weight_Gradient5_Layer1_CH0_256 = process_input_lines(Weight_Gradient5_Layer1_CH0)
    #print("Weight_Gradient5_Layer1_CH0 : ", len(Weight_Gradient5_Layer1_CH0)) 

    Weight_Gradient6_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD5A00,  End_Address=0x8FBD6C00)
    Weight_Gradient6_Layer1_CH0_256 = process_input_lines(Weight_Gradient6_Layer1_CH0)
    #print("Weight_Gradient6_Layer1_CH0 : ", len(Weight_Gradient6_Layer1_CH0)) 

    Weight_Gradient7_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD6C00,  End_Address=0x8FBD7E00)
    Weight_Gradient7_Layer1_CH0_256 = process_input_lines(Weight_Gradient7_Layer1_CH0)
    #print("Weight_Gradient7_Layer1_CH0 : ", len(Weight_Gradient7_Layer1_CH0)) 

    Weight_Gradient8_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD7E00,  End_Address=0x8FBD9000)
    Weight_Gradient8_Layer1_CH0_256 = process_input_lines(Weight_Gradient8_Layer1_CH0)
    #print("Weight_Gradient8_Layer1_CH0 : ", len(Weight_Gradient8_Layer1_CH0)) 

    Weight_Gradient1_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD0000,  End_Address=0x9FBD1200)
    Weight_Gradient1_Layer1_CH1_256 = process_input_lines(Weight_Gradient1_Layer1_CH1)
    #print("Weight_Gradient1_Layer1_CH1 : ", len(Weight_Gradient1_Layer1_CH1)) 

    Weight_Gradient2_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD1200,  End_Address=0x9FBD2400)
    Weight_Gradient2_Layer1_CH1_256 = process_input_lines(Weight_Gradient2_Layer1_CH1)
    #print("Weight_Gradient2_Layer1_CH1 : ", len(Weight_Gradient2_Layer1_CH1)) 

    Weight_Gradient3_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD2400,  End_Address=0x9FBD3600)
    Weight_Gradient3_Layer1_CH1_256 = process_input_lines(Weight_Gradient3_Layer1_CH1)
    #print("Weight_Gradient3_Layer1_CH1 : ", len(Weight_Gradient3_Layer1_CH1)) 

    Weight_Gradient4_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD3600,  End_Address=0x9FBD4800)
    Weight_Gradient4_Layer1_CH1_256 = process_input_lines(Weight_Gradient4_Layer1_CH1)
    #print("Weight_Gradient4_Layer1_CH1 : ", len(Weight_Gradient4_Layer1_CH1)) 

    Weight_Gradient5_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD4800,  End_Address=0x9FBD5A00)
    Weight_Gradient5_Layer1_CH1_256 = process_input_lines(Weight_Gradient5_Layer1_CH1)
    #print("Weight_Gradient5_Layer1_CH1 : ", len(Weight_Gradient5_Layer1_CH1)) 

    Weight_Gradient6_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD5A00,  End_Address=0x9FBD6C00)
    Weight_Gradient6_Layer1_CH1_256 = process_input_lines(Weight_Gradient6_Layer1_CH1)
    #print("Weight_Gradient6_Layer1_CH1 : ", len(Weight_Gradient6_Layer1_CH1)) 

    Weight_Gradient7_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD6C00,  End_Address=0x9FBD7E00)
    Weight_Gradient7_Layer1_CH1_256 = process_input_lines(Weight_Gradient7_Layer1_CH1)
    #print("Weight_Gradient7_Layer1_CH1 : ", len(Weight_Gradient7_Layer1_CH1)) 

    Weight_Gradient8_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD7E00,  End_Address=0x9FBD9000)
    Weight_Gradient8_Layer1_CH1_256 = process_input_lines(Weight_Gradient8_Layer1_CH1)
    #print("Weight_Gradient8_Layer1_CH1 : ", len(Weight_Gradient8_Layer1_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer1_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer1_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer1_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer1_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

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
    Weight_Gradient_Layer1 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer1)]   
    Weight_Gradient_Layer1 = torch.tensor([float(value) for value in Weight_Gradient_Layer1], dtype=torch.float32).reshape(32, 16, 3, 3)   

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()

    #################################################
    #             Backward Layer 0 Start            #
    #################################################
    # check Layer0 IRQ
    check_irq_otherlayer()

    '''
    # Read Gradient of Output After ReLU Backward: 
    Output_Grad1_Layer0_CH0 = Read_DDR(Rd_Address=0x9384000,  End_Address=0x96C4000)
    Output_Grad1_Layer0_CH0 = process_input_lines(Output_Grad1_Layer0_CH0)

    Output_Grad1_Layer0_CH1 = Read_DDR(Rd_Address=0x19384000,  End_Address=0x196C4000)
    Output_Grad1_Layer0_CH1 = process_input_lines(Output_Grad1_Layer0_CH1)

    Output_Grad2_Layer0_CH0 = Read_DDR(Rd_Address=0x96C4000,  End_Address=0x9A04000)
    Output_Grad2_Layer0_CH0 = process_input_lines(Output_Grad2_Layer0_CH0)

    Output_Grad2_Layer0_CH1 = Read_DDR(Rd_Address=0x196C4000,  End_Address=0x19A04000)
    Output_Grad2_Layer0_CH1 = process_input_lines(Output_Grad2_Layer0_CH1)

    Output_Grad3_Layer0_CH0 = Read_DDR(Rd_Address=0x9A04000,  End_Address=0x9D44000)
    Output_Grad3_Layer0_CH0 = process_input_lines(Output_Grad3_Layer0_CH0)

    Output_Grad3_Layer0_CH1 = Read_DDR(Rd_Address=0x19A04000,  End_Address=0x19D44000)
    Output_Grad3_Layer0_CH1 = process_input_lines(Output_Grad3_Layer0_CH1)

    Output_Grad4_Layer0_CH0 = Read_DDR(Rd_Address=0x9D44000,  End_Address=0xA084000)
    Output_Grad4_Layer0_CH0 = process_input_lines(Output_Grad4_Layer0_CH0)

    Output_Grad4_Layer0_CH1 = Read_DDR(Rd_Address=0x19D44000,  End_Address=0x1A084000)
    Output_Grad4_Layer0_CH1 = process_input_lines(Output_Grad4_Layer0_CH1)

    Output_Grad5_Layer0_CH0 = Read_DDR(Rd_Address=0xA084000,  End_Address=0xA3C4000)
    Output_Grad5_Layer0_CH0 = process_input_lines(Output_Grad5_Layer0_CH0)

    Output_Grad5_Layer0_CH1 = Read_DDR(Rd_Address=0x1A084000,  End_Address=0x1A3C4000)
    Output_Grad5_Layer0_CH1 = process_input_lines(Output_Grad5_Layer0_CH1)

    Output_Grad6_Layer0_CH0 = Read_DDR(Rd_Address=0xA3C4000,  End_Address=0xA704000)
    Output_Grad5_Layer0_CH1 = process_input_lines(Output_Grad5_Layer0_CH1)

    Output_Grad6_Layer0_CH1 = Read_DDR(Rd_Address=0x1A3C4000,  End_Address=0x1A704000)
    Output_Grad6_Layer0_CH1 = process_input_lines(Output_Grad6_Layer0_CH1)

    Output_Grad7_Layer0_CH0 = Read_DDR(Rd_Address=0xA704000,  End_Address=0xAA44000)
    Output_Grad7_Layer0_CH0 = process_input_lines(Output_Grad7_Layer0_CH0)

    Output_Grad7_Layer0_CH1 = Read_DDR(Rd_Address=0x1A704000,  End_Address=0x1AA44000)
    Output_Grad7_Layer0_CH1 = process_input_lines(Output_Grad7_Layer0_CH1)

    Output_Grad8_Layer0_CH0 = Read_DDR(Rd_Address=0xAA44000,  End_Address=0xAD84000)
    Output_Grad8_Layer0_CH0 = process_input_lines(Output_Grad8_Layer0_CH0)

    Output_Grad8_Layer0_CH1 = Read_DDR(Rd_Address=0x1AA44000,  End_Address=0x1AD84000)
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
    
    '''


    # Gradient of Beta Calculation:
    # Beta_Gradient_Layer0 = (Output_Grad_Layer0).sum(dim=(0, 2, 3), keepdim=True)

    # Weight Gradient
    Weight_Gradient1_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBD9000,  End_Address=0x8FBD9900)
    Weight_Gradient1_Layer0_CH0_256 = process_input_lines(Weight_Gradient1_Layer0_CH0)
    #print("Weight_Gradient1_Layer0_CH0 : ", len(Weight_Gradient1_Layer0_CH0))   

    Weight_Gradient2_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBD9900,  End_Address=0x8FBDA200)
    Weight_Gradient2_Layer0_CH0_256 = process_input_lines(Weight_Gradient2_Layer0_CH0)
    #print("Weight_Gradient2_Layer0_CH0 : ", len(Weight_Gradient2_Layer0_CH0))    

    Weight_Gradient3_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDA200,  End_Address=0x8FBDAB00)
    Weight_Gradient3_Layer0_CH0_256 = process_input_lines(Weight_Gradient3_Layer0_CH0)
    #print("Weight_Gradient3_Layer0_CH0 : ", len(Weight_Gradient3_Layer0_CH0)) 

    Weight_Gradient4_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDAB00,  End_Address=0x8FBDB400)
    Weight_Gradient4_Layer0_CH0_256 = process_input_lines(Weight_Gradient4_Layer0_CH0)
    #print("Weight_Gradient4_Layer0_CH0 : ", len(Weight_Gradient4_Layer0_CH0)) 

    Weight_Gradient5_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDB400,  End_Address=0x8FBDBD00)
    Weight_Gradient5_Layer0_CH0_256 = process_input_lines(Weight_Gradient5_Layer0_CH0)
    #print("Weight_Gradient5_Layer0_CH0 : ", len(Weight_Gradient5_Layer0_CH0)) 

    Weight_Gradient6_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDBD00,  End_Address=0x8FBDC600)
    Weight_Gradient6_Layer0_CH0_256 = process_input_lines(Weight_Gradient6_Layer0_CH0)
    #print("Weight_Gradient6_Layer0_CH0 : ", len(Weight_Gradient6_Layer0_CH0)) 

    Weight_Gradient7_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDC600,  End_Address=0x8FBDCF00)
    Weight_Gradient7_Layer0_CH0_256 = process_input_lines(Weight_Gradient7_Layer0_CH0)
    #print("Weight_Gradient7_Layer0_CH0 : ", len(Weight_Gradient7_Layer0_CH0)) 

    Weight_Gradient8_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDCF00,  End_Address=0x8FBDD800)
    Weight_Gradient8_Layer0_CH0_256 = process_input_lines(Weight_Gradient8_Layer0_CH0)
    #print("Weight_Gradient8_Layer0_CH0 : ", len(Weight_Gradient8_Layer0_CH0)) 

    Weight_Gradient1_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBD9000,  End_Address=0x9FBD9900)
    Weight_Gradient1_Layer0_CH1_256 = process_input_lines(Weight_Gradient1_Layer0_CH1)
    #print("Weight_Gradient1_Layer0_CH1 : ", len(Weight_Gradient1_Layer0_CH1)) 

    Weight_Gradient2_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBD9900,  End_Address=0x9FBDA200)
    Weight_Gradient2_Layer0_CH1_256 = process_input_lines(Weight_Gradient2_Layer0_CH1)
    #print("Weight_Gradient2_Layer0_CH1 : ", len(Weight_Gradient2_Layer0_CH1)) 

    Weight_Gradient3_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDA200,  End_Address=0x9FBDAB00)
    Weight_Gradient3_Layer0_CH1_256 = process_input_lines(Weight_Gradient3_Layer0_CH1)
    #print("Weight_Gradient3_Layer0_CH1 : ", len(Weight_Gradient3_Layer0_CH1)) 

    Weight_Gradient4_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDAB00,  End_Address=0x9FBDB400)
    Weight_Gradient4_Layer0_CH1_256 = process_input_lines(Weight_Gradient4_Layer0_CH1)
    #print("Weight_Gradient4_Layer0_CH1 : ", len(Weight_Gradient4_Layer0_CH1)) 

    Weight_Gradient5_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDB400,  End_Address=0x9FBDBD00)
    Weight_Gradient5_Layer0_CH1_256 = process_input_lines(Weight_Gradient5_Layer0_CH1)
    #print("Weight_Gradient5_Layer0_CH1 : ", len(Weight_Gradient5_Layer0_CH1)) 

    Weight_Gradient6_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDBD00,  End_Address=0x9FBDC600)
    Weight_Gradient6_Layer0_CH1_256 = process_input_lines(Weight_Gradient6_Layer0_CH1)
    #print("Weight_Gradient6_Layer0_CH1 : ", len(Weight_Gradient6_Layer0_CH1)) 

    Weight_Gradient7_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDC600,  End_Address=0x9FBDCF00)
    Weight_Gradient7_Layer0_CH1_256 = process_input_lines(Weight_Gradient7_Layer0_CH1)
    #print("Weight_Gradient7_Layer0_CH1 : ", len(Weight_Gradient7_Layer0_CH1)) 

    Weight_Gradient8_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDCF00,  End_Address=0x9FBDD800)
    Weight_Gradient8_Layer0_CH1_256 = process_input_lines(Weight_Gradient8_Layer0_CH1)
    #print("Weight_Gradient8_Layer0_CH1 : ", len(Weight_Gradient8_Layer0_CH1)) 

    '''
    test_out = 'Weight_Result/Weight_Gradient1_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient1_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient1_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient2_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient2_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient3_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient3_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient4_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient4_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient5_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient5_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient6_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient6_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient7_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient7_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer0_CH0.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer0_CH0:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()

    test_out = 'Weight_Result/Weight_Gradient8_Layer0_CH1.txt'
    with open(test_out, 'w+') as test_output:
        for item in Weight_Gradient8_Layer0_CH1:
            line = str(item) 
            test_output.write(line + '\n')   
    test_output.close()
    '''

    Weight_Gradient1_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient1_Layer0_CH0_256, Weight_Gradient1_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient2_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient2_Layer0_CH0_256, Weight_Gradient2_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient3_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient3_Layer0_CH0_256, Weight_Gradient3_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient4_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient4_Layer0_CH0_256, Weight_Gradient4_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient5_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient5_Layer0_CH0_256, Weight_Gradient5_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient6_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient6_Layer0_CH0_256, Weight_Gradient6_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient7_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient7_Layer0_CH0_256, Weight_Gradient7_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient8_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient8_Layer0_CH0_256, Weight_Gradient8_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
    Weight_Gradient_Layer0 = [Weight_Gradient1_Layer0, Weight_Gradient2_Layer0, Weight_Gradient3_Layer0, Weight_Gradient4_Layer0, Weight_Gradient5_Layer0, 
                            Weight_Gradient6_Layer0, Weight_Gradient7_Layer0, Weight_Gradient8_Layer0]
    Weight_Gradient_Layer0 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer0)]   
    Weight_Gradient_Layer0 = torch.tensor([float(value) for value in Weight_Gradient_Layer0], dtype=torch.float32).reshape(16, 3, 3, 3)   

     # Gradient Value for Weight Update
    Weight_Gradient = [Weight_Gradient_Layer0, Weight_Gradient_Layer1, Weight_Gradient_Layer2, Weight_Gradient_Layer3, Weight_Gradient_Layer4,
                       Weight_Gradient_Layer5, Weight_Gradient_Layer6, Weight_Gradient_Layer7, Weight_Gradient_Layer8]
    
    Beta_Gradient = [dL_dbeta_0, dL_dbeta_1, dL_dbeta_2, dL_dbeta_3, dL_dbeta_4, dL_dbeta_5,
                     dL_dbeta_6, dL_dbeta_7]
    
    Gamma_Gradient = [dL_dgamma_0, dL_dgamma_1, dL_dgamma_2, dL_dgamma_3, dL_dgamma_4,
                      dL_dgamma_5, dL_dgamma_6, dL_dgamma_7]

    d = Device("0000:08:00.0")
    bar = d.bar[0]

    resume()

    print("Backward Done")
    
def Write_Weight(epoch):
    # Pre-Processing Class Initialization
    global Weight_Dec_List, Weight_List, Bias_Dec_List, Bias_List, Beta_Dec_List, Gamma_Dec_List, Gamma_List, Beta_List
    
    PreProcessing = Pre_Processing(Mode                 =   Mode,
                                Brain_Floating_Point =   Brain_Floating_Point,
                                Exponent_Bits        =   Exponent_Bits,
                                Mantissa_Bits        =   Mantissa_Bits)   
    if Bias_Converted:
        Bias_Dec_List, Bias_List = PreProcessing.Bias_Converted_Func()
    
    if BN_Param_Converted:
        Beta_Dec_List, Beta_List = PreProcessing.Beta_Param_Converted_Func()
        Gamma_Dec_List, Gamma_List = PreProcessing.Gamma_Param_Converted_Func()
        RunningMean_Dec_List, RunningMean_List = PreProcessing.Running_Mean_Param_Converted_Func()
        RunningVar_Dec_List, RunningVar_List = PreProcessing.Running_Var_Param_Converted_Func()
        
    if Weight_Converted:
        if epoch == 0:
            print("\t --> " + " The Weights are processing, Please Wait for a Moment!")
            Weight_Dec_List, Weight_List = PreProcessing.Weight_Converted_Func()
        else:
            Weight_Dec_List, Weight_List = PreProcessing.Weight_Converted_Func()
            Weight_1st_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, Weight_List[0], ['0000']*16, ['0000']*16, ['0000']*16, Iteration="1")
            Weight_1st_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, Weight_List[1], ['0000']*32, ['0000']*32, ['0000']*32, Iteration="1")
            Weight_1st_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, Weight_List[2], ['0000']*64, ['0000']*64, ['0000']*64, Iteration="1")
            Weight_1st_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, Weight_List[3], ['0000']*128, ['0000']*128, ['0000']*128, Iteration="1")
            Weight_1st_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, Weight_List[4], ['0000']*256, ['0000']*256, ['0000']*256, Iteration="1")
            Weight_1st_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, Weight_List[5], ['0000']*512, ['0000']*512, ['0000']*512, Iteration="1")
            Weight_1st_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, Weight_List[6], ['0000']*1024, ['0000']*1024, ['0000']*1024, Iteration="1")
            Weight_1st_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], ['0000']*1024, ['0000']*1024, ['0000']*1024, Iteration="1")
            Weight_1st_Layer8 = New_Weight_Hardware_ReOrdering_Layer8(128, 1024, Weight_List[8], Bias_List)
            
            # List for Each DDR Channels: 
            Weight_1st_CH0 = Weight_1st_Layer0[0] + Weight_1st_Layer1[0] + Weight_1st_Layer2[0] + Weight_1st_Layer3[0] + Weight_1st_Layer4[0] + \
                            Weight_1st_Layer5[0] + Weight_1st_Layer6[0] + Weight_1st_Layer7[0] + Weight_1st_Layer8[0]
            Weight_1st_CH1 = Weight_1st_Layer0[1] + Weight_1st_Layer1[1] + Weight_1st_Layer2[1] + Weight_1st_Layer3[1] + Weight_1st_Layer4[1] + \
                            Weight_1st_Layer5[1] + Weight_1st_Layer6[1] + Weight_1st_Layer7[1] + Weight_1st_Layer8[1]
            
            # Break 256To32 and Flip the Data: 
            Weight_1st_CH0 = Break_FlipHex_256To32(Weight_1st_CH0)
            Weight_1st_CH1 = Break_FlipHex_256To32(Weight_1st_CH1)
                            
            # Write Weights into DDR: 
            Write_DDR(Weight_1st_CH0, Wr_Address=0x80000000)
            Write_DDR(Weight_1st_CH1, Wr_Address=0x88000000)    

def check_irq_layer0():
    input_file_name = "/proc/interrupts"
    output_file_name_1 = "interrupt.txt"
    output_file_name_2 = "interrupt_old.txt"
    irq_val=0
    while irq_val == 0:                        
        if os.path.isfile(output_file_name_2):

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
                            print("interrupt1: 1")
                            irq_val = 1
                            # self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)


                    # if irq_val != 1:
                    #     print("layer0 interrupt1: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    # print("Done")
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
                            # self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")

                            print("interrupt: 1")
                        # else:
                            # irq_val=0
                            # print("layer0 interrupt0: 0") 

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

def check_irq_otherlayer():
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
                        print("interrupt: 1")
                        irq_val = 1
                        # L1_IRQ_canvas.itemconfig(L1_IRQ, fill="green")
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                # if irq_val != 1:
                #     print("layer1 interrupt: 0")

                with open(output_file_name_1, "rb") as file1, \
                    open(output_file_name_2, "wb") as file2:

                    buffer = file1.read(MAX_LINE_LENGTH)
                    while buffer:
                        file2.write(buffer)
                        buffer = file1.read(MAX_LINE_LENGTH)

                # print("Done")
                file1.close()
                file2.close()
        
        # print("extract xdma line Done!\n")     

def resume():
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    # resume    
    print("Resume Process")
    bar.write(0x20, 1)

    bar.write(0x20, 0)

def flip_8_line(data):
    for i in range(0, len(data), 8):
        group = data[i:i+8]
        yield from reversed(group)
    
def data_32_to_16(data):
    data = flip_8_line(data)
    hex_values = (hex(value)[2:].upper().zfill(8) for value in data)
    hex_strings = ''.join(hex_values)
    formatted_data = [hex_strings[i:i+4] for i in range(0, len(hex_strings), 4)]
    return formatted_data

if __name__ == "__main__":
    app = App()
    app.mainloop()

