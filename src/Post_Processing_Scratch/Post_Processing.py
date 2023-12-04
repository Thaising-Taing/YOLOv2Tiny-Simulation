from .Post_Processing_Function import *
from .Loss import *
from .Dataset import *
import torch

# from Calculate_Loss import *
# from Dataset import *
# from Post_Processing_Function import *


# 2023/06/21: Implemented By Thaising
# Combined Master-PhD in MSISLAB


# Conditions for Reading Original_Data from Hardware Processing:
class Post_Processing:
    def __init__(self, Mode, Brain_Floating_Point, Exponent_Bits, Mantissa_Bits, OutImage_Data_CH0, OutImage_Data_CH1):
        self.Mode = Mode
        self.Brain_Floating_Point = Brain_Floating_Point
        self.Exponent_Bits = Exponent_Bits
        self.Mantissa_Bits = Mantissa_Bits
        self.OutImage_Data_CH0 = OutImage_Data_CH0
        self.OutImage_Data_CH1 = OutImage_Data_CH1

    def PostProcessing(self):
        # Creating the Directories
        global Layer8_Loss_Gradient, Numerical_Loss
        if self.Brain_Floating_Point:
            Output_Image = Read_OutFmap_Layer8_BFPtoDec(self.OutImage_Data_CH0, self.OutImage_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            # Output_Image = Output_Image[0:(1*125*13*13)]
        else:
            Output_Image = Read_OutFmap_Layer8_FP32toDec(self.OutImage_Data_CH0, self.OutImage_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)

        # Mode is Training
        if self.Mode == "Training":
            print("\t === Our Post-Processing is finished! ===\n")
            # Convert Output Image into List of Floating 32 Format
            Float_OutputImage = [np.float32(x) for x in Output_Image]
            # Loss Calculation and Loss Gradient Calculation
            Layer8_Loss, Layer8_Loss_Gradient = loss(Float_OutputImage, gt_boxes=gt_boxes,
                                                     gt_classes=gt_classes, num_boxes=num_boxes)
            Numerical_Loss = Numerical_Loss(Layer8_Loss)
            print("\t === The Backward is starting! ===")
            # Pass Back the Original_Data into Hardware Processing

            # Mode is Inference
        if self.Mode == "Inference":
            print("\t === Our Post-Processing is finished! ===\n")
            # Convert Output Image into List of Floating 32 Format
            # Float_OutputImage = [np.float32(x) for x in Output_Image]
            # Loss Calculation and Loss Gradient Calculation
            Layer8_Loss, Layer8_Loss_Gradient = loss(self.OutImage_Data, gt_boxes=gt_boxes,
                                                     gt_classes=gt_classes, num_boxes=num_boxes)
            Numerical_Loss = Numerical_Loss(Layer8_Loss)
            print("\t === The Backward is starting! ===")

        return Numerical_Loss, Layer8_Loss_Gradient


# if __name__ == "__main__":
#     PostProcessing = Post_Processing(
#         Mode="Training",
#         Brain_Floating_Point=True,
#         Exponent_Bits=8,
#         Mantissa_Bits=23
#     )
#
#     Numerical_Loss, Layer8_Loss_Gradient = PostProcessing.PostProcessing()
#     print(Numerical_Loss)
#     print(Layer8_Loss_Gradient)

