from pypcie import Device
import os

MAX_LINE_LENGTH = 1000

def check_irq_layer0():
    input_file_name = "/proc/interrupts"
    output_file_name_1 = "src/GiTae/interrupt.txt"
    output_file_name_2 = "src/GiTae/interrupt_old.txt"
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
                            # if DEBUG: print("interrupt1: 1")
                            irq_val = 1
                            # self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)


                    # if irq_val != 1:
                    #     if DEBUG: print("layer0 interrupt1: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    # if DEBUG: print("Done")
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

                        #     if DEBUG: print("interrupt: 1")
                        # else:
                        #     irq_val=0
                        #     if DEBUG: print("layer0 interrupt0: 0") 

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
 

# print("Button 3 clicked")
d = Device("0000:08:00.0")
bar = d.bar[0]
#self.textbox.insert("0.0", "CTkTextbox\n\n" )

#microcode = Microcode("mic_2iteration_forward_hex_add_0x.txt") 
microcode = Microcode("src/GiTae/MICROCODE_FORWARD.txt")

for i in range (0, len(microcode)):
    bar.write(0x4, microcode[i]) # wr mic
    bar.write(0x8, i) # wr addr
    bar.write(0x0, 0x00000012) # wr en
    bar.write(0x0, 0x00000010) # wr en low
print("mic write done")  

        
d = Device("0000:08:00.0")
bar = d.bar[0]
bar.write(0x0, 0x00000011) # yolo start
bar.write(0x0, 0x00000010) # yolo start low
bar.write(0x8, 0x00000011) # rd addr
bar.write(0x0, 0x00000014) # rd en
bar.write(0x0, 0x00000010) # rd en low
bar.write(0x18, 0x00008001) # axi addr
bar.write(0x14, 0x00000001) # axi rd en
bar.write(0x14, 0x00000000) # axi rd en low
print("start")
check_irq_layer0()
print("end")