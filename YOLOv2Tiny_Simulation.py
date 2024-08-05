import os
import sys
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
import shutil
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import tqdm
import warnings
from datetime import datetime
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
from Weight_Update_Algorithm.new_weight_update import new_weight_update, new_weight_update_two, sgd_momentum_update
# from Weight_Update_Algorithm.new_weight_update import initial_lr as Initial_LR_SGD

import numpy as np
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
from Thaising_PyTorch import TorchSimulation_LN
from Thaising_PyTorch_BatchNorm import TorchSimulation_BN
from Thaising_Python import PythonSimulation
from Wathna_pytorch import Pytorch
from Wathna_python import Python
from batchnorm_python import Python_bn
from batchnorm_pytorch import Pytorch_bn
from Weight_Update_Algorithm.new_weight_update import new_weight_update
import checkmap_new

torch.manual_seed(3407)
np.random.seed(3407)

DDR_SIZE = 0x10000
MAX_LINE_LENGTH = 1000
save_debug_data = False

try: from colorama import Fore, Back, Style
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'colorama'])    
    from colorama import Fore, Back, Style

class YOLOv2Tiny_Simulation():

    def __init__(self, Type):
        super().__init__()

        # Pre-define Conditions:
        self.Mode = "Training"  # Declare whether is Training or Inference

        # Running Hardware Model:
        self.YOLOv2_Hardware_Forward     = True
        self.YOLOv2_Hardware_Backward    = True
        
        # Floating Point Parameters that we use
        self.FloatingPoint_Format = "FP32"

        self.Selected_FP_Format = {
            "FP32": (8, 23),
            "Bfloat16": (5, 10),
            "Custom": (7, 8)
        }

        self.Exponent_Bits, self.Mantissa_Bits = self.Selected_FP_Format.get(self.FloatingPoint_Format, (0, 0))

        # Pre-Processing Pre-Defined Conditions
        self.Brain_Floating_Point    = True  # Declare Whether Bfloat16 Conversion or not

        self.phase_Forward = 'Forward'
        self.phase_backward = 'Backward'
        
        self.bestmAP=0
        self.bestmAPepoch=0
        self.Loss_Val = 1000
        
        self.mode = Type
        
    def Run_Train(self):  
        self.Show_Text(f"\nStart Training", clr=Fore.MAGENTA)
        self.Show_Text(f"Mode                       : {Fore.LIGHTYELLOW_EX}{self.mode}\n", clr=Fore.BLUE)
        self.stop_process = False
        
        self.parse_args()
        self.Pre_Process()
        self.Load_Weights()
        self.Load_Dataset()
        self.Create_Output_Dir()

        if 'voc' in self.imdb_train_name[18:]:  self.args.output_dir = os.path.join( self.args.output_dir , 'FullData' )
        else:                                   self.args.output_dir = os.path.join( self.args.output_dir , self.imdb_train_name[18:] )

        self.Show_Text(f"Output directory           : {Fore.LIGHTYELLOW_EX}{self.args.output_dir}\n"                 , clr=Fore.BLUE)
        self.Show_Text(f'Training Dataset           : {Fore.LIGHTYELLOW_EX}{self.imdb_train_name}'                  , clr=Fore.BLUE)
        self.Show_Text(f'Number of training images  : {Fore.LIGHTYELLOW_EX}{len(self.train_dataset._image_paths)}'   , clr=Fore.BLUE)
        self.Show_Text(f'Batch size                 : {Fore.LIGHTYELLOW_EX}{self.args.batch_size}'                  , clr=Fore.BLUE)
        print()
        
        # Loop for total number of epochs
        _full_dataset_loop = tqdm(range(self.args.start_epoch, self.args.max_epochs), total=self.args.max_epochs   ,   leave=False)
        for self.epoch in range(self.args.start_epoch, self.args.max_epochs):
                _full_dataset_loop.set_description(   f"{Fore.GREEN+Style.BRIGHT}Epoch {self.epoch+1}/{self.args.max_epochs}")
                if self.stop_process: break
                
                # Data iterator
                self.data_iter = iter(self.train_dataloader)
                
                # Loop for current epoch - all batches
                _current_epoch_loop = tqdm(range(self.iters_per_epoch_train),leave=False)


            

                for _batch, step in enumerate(_current_epoch_loop):
                    if self.stop_process: break
                    _current_epoch_loop.set_description(  f"    {Fore.LIGHTGREEN_EX}Epoch {self.epoch}{Style.RESET_ALL} - Batch {_batch} - Loss {self.Loss_Val}")
                    
                    # Loading dataset
                    self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.data_iter)
                    
                    # Perform Training
                    self.Before_Forward() ######################### - Individual Functions
                    self.Forward() ################################ - Individual Functions
                    # self.Visualize()
                    self.Calculate_Loss()
                        
                    self.Before_Backward() ######################## - Individual Functions
                    self.Backward() ############################### - Individual Functions
                    self.Weight_Update(self.epoch)
                    
                # self.Check_mAP()
                self.Save_Weights(self.epoch)
        #     self.Save_Pickle()
        self.Post_Epoch()
        self.Show_Text(f"Training is finished", clr=Fore.BLUE)

    def Run_Infer(self):
        self.stop_process = False
        
        self.Show_Text(f"Start Inference", clr=Fore.MAGENTA)
        self.parse_args()
        self.Pre_Process()
        self.Create_Output_Dir()
        self.Load_Weights()
        self.Load_Dataset()

        self.whole_process_start = time.time()
        self.data_iter = iter(self.test_dataloader)
        
        for step in tqdm(range(self.iters_per_epoch_test), desc=f"Inference", total=self.iters_per_epoch_test):
            self.im_data, self.gt_boxes, self.gt_classes, self.num_obj = next(self.data_iter)
            
            self.batch = step
            self.Before_Forward() ######################### - Individual Functions
            self.Forward_Infer() ################################ - Individual Functions
            self.Visualize()
            # self.Visualize_All()
              
        self.Show_Text(f"Total Images with detections   : {self.count['detections']}", clr=Fore.LIGHTGREEN_EX)
        self.Show_Text(f"Total Images without detections: {self.count['no_detections']}", clr=Fore.LIGHTGREEN_EX)
        self.Show_Text(f"Inference is finished.", clr=Fore.MAGENTA)
        
    def Run_Validation(self):
        self.stop_process = False
        self.Show_Text(f"Start validation", clr=Fore.MAGENTA)
        self.parse_args()
        self.Pre_Process()
        self.Create_Output_Dir()
        self.Load_Weights()

        self.Check_mAP()
        self.Show_Text(f"Validation is finished", clr=Fore.BLUE)
        
    # Training Helper Functions
    def Save_File(self, _path, data):
        # _dir = _path.split('/')[1:-1]
        # if len(_dir)>1: _dir = os.path.join(_dir)
        # else: _dir = _dir[0]
        # if not os.path.isdir(_dir): os.mkdir(_dir)
        with open(_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def Load_File(self, _path):
        _dir = _path.split('/')[1:-1]
        if len(_dir)>1: _dir = os.path.join(_dir)
        else: _dir = _dir[0]
        if not os.path.isdir(_dir): os.mkdir(_dir)
        with open(_path, 'rb') as handle:
            b = pickle.load(handle)
        return b
    
    def parse_args(self):
        """
        Parse input arguments
        """
        parser = argparse.ArgumentParser(description='Yolo v2')
        parser.add_argument('--max_epochs', dest='max_epochs',
                            help='number of epochs to train',
                            default=300, type=int)
        parser.add_argument('--start_epoch', dest='start_epoch',
                            default=0, type=int)
        parser.add_argument('--total_training_set', dest='total_training_set',
                            default=256, type=int)
        parser.add_argument('--total_inference_set', dest='total_inference_set',
                            default=64, type=int)
        parser.add_argument('--batch_size', dest='batch_size',
                            default=8, type=int)
        parser.add_argument('--nw', dest='num_workers',
                            help='number of workers to load training data',
                            default=16, type=int)
        parser.add_argument('--use_small_dataset', dest='use_small_dataset',
                            default=True, type=bool)
        parser.add_argument('--save_interval', dest='save_interval',
                            default=10, type=int)
        parser.add_argument('--dataset', dest='dataset', default="full", 
                            const='all', type=str, nargs='?',
                            choices=['full', 'car', 'car-64', 'random-64', 'random-128', 'random-256', 'random-512', 'random-5517'],
                            help='list servers, storage, or both (default: %(default)s)')
        parser.add_argument('--pretrained', dest='pretrained',
                            default="Dataset/Dataset/pretrained/scratch.pth", type=str)
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
        
        if self.mode == "PytorchSim_LN":  self.TorchSimulation_LN   = TorchSimulation_LN(self)
        if self.mode == "PytorchSim_BN":  self.TorchSimulation_BN   = TorchSimulation_BN(self)  
        if self.mode == "PythonSim"    :  self.PythonSimulation     = PythonSimulation(self)

    def Create_Output_Dir(self):
        if self.mode == "PythonSim"         :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        elif self.mode == "PytorchSim_LN"     :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        elif self.mode == "PytorchSim_BN"     :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        else                                  :  self.args.output_dir = self.args.output_dir + '/' + self.mode
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

    def Load_Weights(self):
        s = time.time()
        self.loaded_weights = self.Load_Weight_Simulation(_path = self.args.pretrained)
        
        if self.mode == "PythonSim"   :  self.PythonSimulation.load_weights(self.loaded_weights)
        if self.mode == "PytorchSim_LN"  :  self.TorchSimulation_LN.load_weights(self.loaded_weights)
        if self.mode == "PytorchSim_BN"  :  self.TorchSimulation_BN.load_weights(self.loaded_weights)
        
        e = time.time()

    def update_weights(self, data):
        [ self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, self.Running_Mean_Dec, self.Running_Var_Dec ] = data

    def Load_Dataset(self):
        # Remove previous cache
        if os.path.isdir('data/cache'): shutil.rmtree("data/cache")
        # -------------------------------------- Car - Dataset -----------------------------------------------------
        # ------------ Train Images
        if self.args.dataset=='full'       : self.imdb_train_name = 'voc_2007_trainval+voc_2012_trainval'
        if self.args.dataset=='car'        : self.imdb_train_name = 'voc_2007_trainval-car'
        if self.args.dataset=='car-64'     : self.imdb_train_name = 'voc_2007_trainval-car-64'
        if self.args.dataset=='random-64'  : self.imdb_train_name = 'voc_2007_trainval-random-64'
        if self.args.dataset=='random-128' : self.imdb_train_name = 'voc_2007_trainval-random-128' 
        if self.args.dataset=='random-256' : self.imdb_train_name = 'voc_2007_trainval-random-256'
        if self.args.dataset=='random-512' : self.imdb_train_name = 'voc_2007_trainval-random-512'
        if self.args.dataset=='random-5517': self.imdb_train_name = 'voc_2012_train-5517'
        
        self.train_dataset              = self.get_dataset(self.imdb_train_name)
        
        if self.args.num_workers==1:
            import torchdata.datapipes.iter as pipes
            pipe = pipes.InMemoryCacheHolder(self.train_dataset, size=32000).sharding_filter() # 8GB
            self.train_dataloader = DataLoader(     pipe, 
                                                    batch_size=self.args.batch_size, 
                                                    shuffle=False,
                                                    num_workers=self.args.num_workers, 
                                                    collate_fn=detection_collate, 
                                                    drop_last=True,
                                                    persistent_workers=True, 
                                                    pin_memory=True,    
                                                    # prefetch_factor=2                                                       
                                                )
        else:
            self.train_dataloader = DataLoader(     self.train_dataset, 
                                                    batch_size=self.args.batch_size, 
                                                    shuffle=False,
                                                    num_workers=self.args.num_workers, 
                                                    collate_fn=detection_collate, 
                                                    drop_last=True,
                                                    persistent_workers=True, 
                                                    pin_memory=True,    
                                                    # prefetch_factor=2                                                       
                                                )
            
        self.iters_per_epoch_train      = int(len(self.train_dataset) / self.args.batch_size)
        

        # ------------ Test Images
        self.imdb_test_name                 = 'voc_2007_test-car'
        self.test_dataset                   = self.get_dataset(self.imdb_test_name)
        self.test_dataloader                = DataLoader(  self.test_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=True, 
                                                            num_workers=self.args.num_workers, 
                                                            collate_fn=detection_collate, 
                                                            drop_last=True,
                                                            persistent_workers=True,                                                            
                                                        )
        self.iters_per_epoch_test           = int(len(self.test_dataset) / self.args.batch_size)
        
        # -------------------------------------- Full - Dataset -----------------------------------------------------
        # ------------ Train Images
        self.imdb_train_name_full           = 'voc_2007_trainval+voc_2012_trainval'
        self.train_dataset_full             = self.get_dataset(self.imdb_train_name_full)
        self.train_dataloader_full          = DataLoader(   self.train_dataset_full, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=False,
                                                            num_workers=self.args.num_workers, 
                                                            collate_fn=detection_collate, 
                                                            drop_last=True,
                                                            persistent_workers=True,                                                            
                                                        )
        self.iters_per_epoch_train_full     = int(len(self.train_dataset_full) / self.args.batch_size)
        
        # ------------ Full Test Images
        self.imdb_test_name                 = 'voc_2007_test'
        self.test_dataset                   = self.get_dataset(self.imdb_test_name)
        self.test_dataloader                = DataLoader(   self.test_dataset, 
                                                            batch_size=self.args.batch_size, 
                                                            shuffle=True, 
                                                            num_workers=self.args.num_workers, 
                                                            collate_fn=detection_collate, 
                                                            drop_last=True,
                                                            persistent_workers=True,                                                            
                                                        )
        self.iters_per_epoch_test           = int(len(self.test_dataset) / self.args.batch_size)
        
    def Adjust_Learning_Rate(self):
        # Various of Learning will Change with the Epochs
        # combined_epoch = epoch_num +  ( self.epoch* self.iters_per_epoch_train )
        global_epoh = self.epoch
        # local_epoch = epoch_num
        if global_epoh in cfg.decay_lrs:
            self.Shoaib.custom_optimizer.param_groups[0]['lr'] =  cfg.decay_lrs[global_epoh]
            # self.Show_Text('Learning Rate is adjusted to: ' + str(self.Shoaib.custom_optimizer.param_groups[0]['lr']), clr=Fore.MAGENTA, end='\r')
            
    def Before_Forward(self):
        if self.mode == "PythonSim"    :  pass
        if self.mode == "PytorchSim_LN"   :  pass
        if self.mode == "PytorchSim_BN"   :  pass

    def Forward(self):
        if self.mode == "PythonSim"    :  self.PythonSimulation.Forward(self)
        if self.mode == "PytorchSim_LN"   :  self.TorchSimulation_LN.Forward(self)
        if self.mode == "PytorchSim_BN"   :  self.TorchSimulation_BN.Forward(self)
    
    def Forward_Infer(self):
        if self.mode == "PythonSim"    :  self.PythonSimulation.Forward(self)
        if self.mode == "PytorchSim_LN"   :  self.TorchSimulation_LN.Forward(self)
        if self.mode == "PytorchSim_BN"   :  self.TorchSimulation_BN.Forward(self)
    
    def Calculate_Loss(self):
        if self.mode == "PythonSim"    :  self.PythonSimulation.Calculate_Loss(self)
        if self.mode == "PytorchSim_LN"   :  self.TorchSimulation_LN.Calculate_Loss(self)
        if self.mode == "PytorchSim_BN"   :  self.TorchSimulation_BN.Calculate_Loss(self)

        
        if self.mode == "PythonSim"    :  _data =  self.PythonSimulation
        if self.mode == "PytorchSim_LN":  _data =  self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN":  _data =  self.TorchSimulation_BN
        self.Loss_Val = _data.loss
        
        if np.isnan(self.Loss_Val.detach().cpu().numpy()):
            print("Not a number")
            import pdb
            pdb.set_trace()

    def Before_Backward(self):
        if self.mode == "PythonSim"    :  pass
        if self.mode == "PytorchSim_LN"   :  pass
        if self.mode == "PytorchSim_BN"   :  pass
        
    def Backward(self):
        if self.mode == "PythonSim"    :  self.PythonSimulation.Backward(self)
        if self.mode == "PytorchSim_LN"   :  self.TorchSimulation_LN.Backward(self)
        if self.mode == "PytorchSim_BN"   :  self.TorchSimulation_BN.Backward(self)

    def Register_loaded_parms(self, loaded_weights):

        if self.mode == "PythonSim"    :  _data =  self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data =  self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data =  self.TorchSimulation_BN
        
        _data.Weight, _data.Bias, _data.Gamma_WeightBN, _data.BetaBN, _data.Running_Mean_Dec, _data.Running_Var_Dec = loaded_weights

    def Weight_Update(self, epochs):
        if self.mode == "PythonSim"    :  _data =  self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data =  self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data =  self.TorchSimulation_BN

        new_weights, optims = sgd_momentum_update(Inputs = [_data.Weight,  _data.Bias,  _data.Gamma,  _data.Beta],
                                        gInputs = [_data.gWeight, _data.gBias, _data.gGamma, _data.gBeta], \
                                            epochs = epochs, optimizer_config=_data.optimizer_config)
        
        _data.Weight, _data.Bias, _data.Gamma, _data.Beta = new_weights
        _data.optimizer_config = optims
        
        if self.mode == "PythonSim"    :  self.PythonSimulation.load_weights(new_weights)
        if self.mode == "PytorchSim_LN"   :  self.TorchSimulation_LN.load_weights(new_weights)
        if self.mode == "PytorchSim_BN"   :  self.TorchSimulation_BN.load_weights(new_weights)
            
        [self.Weight, self.Bias, self.Gamma, self.Beta] = new_weights
    
    def Save_Weights(self, epoch):
        if self.mode == "PythonSim"    :  _data = self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data = self.TorchSimulation_BN

        if self.mode in ["PytorchSim_BN", "PytorchSim_LN", "PytorchSim_BN"]:
            Simulation_Weight = {
                'W0': _data.Weight[0],
                'W1': _data.Weight[1],
                'W2': _data.Weight[2],
                'W3': _data.Weight[3],
                'W4': _data.Weight[4],
                'W5': _data.Weight[5],
                'W6': _data.Weight[6],
                'W7': _data.Weight[7],
                'W8': _data.Weight[8],
                'b8': _data.Bias,
                'gamma0': _data.Gamma[0],
                'gamma1': _data.Gamma[1],
                'gamma2': _data.Gamma[2],
                'gamma3': _data.Gamma[3],
                'gamma4': _data.Gamma[4],
                'gamma5': _data.Gamma[5],
                'gamma6': _data.Gamma[6],
                'gamma7': _data.Gamma[7],
                'beta0': _data.Beta[0],
                'beta1': _data.Beta[1],
                'beta2': _data.Beta[2],
                'beta3': _data.Beta[3],
                'beta4': _data.Beta[4],
                'beta5': _data.Beta[5],
                'beta6': _data.Beta[6],
                'beta7': _data.Beta[7],
                'running_mean0': _data.Running_Mean_Dec[0],
                'running_mean1': _data.Running_Mean_Dec[1],
                'running_mean2': _data.Running_Mean_Dec[2],
                'running_mean3': _data.Running_Mean_Dec[3],
                'running_mean4': _data.Running_Mean_Dec[4],
                'running_mean5': _data.Running_Mean_Dec[5],
                'running_mean6': _data.Running_Mean_Dec[6],
                'running_mean7': _data.Running_Mean_Dec[7],
                'running_var0': _data.Running_Var_Dec[0],
                'running_var1': _data.Running_Var_Dec[1],
                'running_var2': _data.Running_Var_Dec[2],
                'running_var3': _data.Running_Var_Dec[3],
                'running_var4': _data.Running_Var_Dec[4],
                'running_var5': _data.Running_Var_Dec[5],
                'running_var6': _data.Running_Var_Dec[6],
                'running_var7': _data.Running_Var_Dec[7]
            }
            output_dir = self.args.output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': Simulation_Weight
            }, save_name)

    def Check_mAP_new(self):

        if self.mode == "PythonSim"    :  _data = self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data = self.TorchSimulation_BN

        Inputs_with_running = _data.Weight, _data.Bias, _data.Gamma, _data.Beta, _data.Running_Mean_Dec, _data.Running_Var_Dec
        
    
    def Check_mAP(self):
        if self.mode == "PythonSim"         :  _data = self.PythonSimulation
        if self.mode == "PytorchSim_LN"     :  _data = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"     :  _data = self.TorchSimulation_BN
        
        _w = _data.Weight, _data.Bias, _data.Gamma, _data.Beta, _data.Running_Mean_Dec, _data.Running_Var_Dec
        mAP = checkmap_new.check( weights = _w, args=self.args, mode = self.mode, _data=_data)
        
        mAP_file = self.args.output_dir + '/mAP.txt'
        with open(mAP_file, mode="a+") as output_file_1:
            output_file_1.write(f"{mAP} \n")
                
    def Post_Epoch(self): 
        if self.mode == "Pytorch_LN"      :  _data =  self.Pytorch
        if self.mode == "Python"       :  _data =  self.Python
        if self.mode == "Pytorch_BN"   :  _data =  self.Pytorch_bn
        if self.mode == "Python_BN"    :  _data =  self.Python_bn
        if self.mode == "PythonSim"    :  _data =  self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data =  self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data =  self.TorchSimulation_BN
        if self.mode == "PythonCUDA"   :  _data =  self.CUDA32
        if self.mode == "PythonCUDA16" :  _data =  self.CUDA16
        if self.mode == "RFFP_CUDA"    :  _data =  self.RFFP_CUDA
        if self.mode == "FPGA"         :  _data =  self.FPGA
        
        # self.whole_process_end = time.time()
        # self.whole_process_time = self.whole_process_end - self.whole_process_start
        self.output_text = f"Epoch: {self.epoch+1}/{self.args.max_epochs}--Loss: {_data.Loss}"
        self.Show_Text(self.output_text)
    
    def Visualize(self):
        if self.mode == "PythonSim"    :  _data = self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data = self.TorchSimulation_BN
        
        out_batch = _data.out
        
        self.Show_Text(f"Infer - {self.mode} - {out_batch.shape}", clr=Fore.BLUE)
        
        for i, (img,out) in enumerate(zip(_data.image,out_batch)):
            _img = img.cpu().detach().numpy().astype(np.uint8)
            _img = np.transpose(_img, (1,2,0))
            
            im_info = dict()
            im_info['height'], im_info['width'], _  = _img.shape
            
            yolo_output = self.reshape_outputs(out)
            yolo_output = [item[0].data for item in yolo_output]
                
            # detections = yolo_eval(yolo_output, im_info, conf_threshold=0.2, nms_threshold=0.5)
            detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)
            
            if len(detections) > 0:
                det_boxes = detections[:, :5].cpu().numpy()
                det_classes = detections[:, -1].long().cpu().numpy()
                
                temp_image_path = 'Output/temp.jpg'
                plt.imsave(temp_image_path, _img)
                img = Image.open(temp_image_path)
                im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=self.classes)
                        
                self.count['detections']+=1
                self.Show_Text(f"{len(detections)} Detections", clr=Fore.BLUE)
                
                plt.figure(f'Output Image')
                plt.imshow(im2show)
                plt.show(block=True)
                
            else:
                self.count['no_detections']+=1
                self.Show_Text(f"No Detections", clr=Fore.BLUE)
                # self.Show_Text(f"Batch {self.batch} - Image {i+1} -- No Detections", end='')

    def Visualize_All(self):
        if self.mode == "PythonSim"    :  _data = self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data = self.TorchSimulation_BN
        
        out_batch_torch  = self.Load_File("output_of_Forward_Torch.pickle")
        
        out_batch_sim    = self.Load_File("output_of_Forward_sim.pickle")
        
        out_batch_fpga   = self.Load_File("output_of_Forward_FPGA.pickle")
        
        
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
        if self.mode == "PythonSim"    :  _data = self.PythonSimulation
        if self.mode == "PytorchSim_LN"   :  _data = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN"   :  _data = self.TorchSimulation_BN
        
        out_batch = _data.out # Out of Forward
        
        self.Show_Text(f"Validate - {self.mode} - {self.batch} - {out_batch.shape}")
        
        for i, (img,out) in enumerate(zip(_data.image,out_batch)):
            self.img_id += 1
            _img = img.cpu().detach().numpy().astype(np.uint8)
            _img = np.transpose(_img, (1,2,0))
            
            im_info = dict()
            im_info['height'], im_info['width'], _  = _img.shape
            
            yolo_output = self.reshape_outputs(out)
            yolo_output = [item[0].data for item in yolo_output]
                
            # detections = yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4)
            detections = yolo_eval(yolo_output, im_info, conf_threshold=0.005, nms_threshold=0.45)
            
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

    def Show_Text(self, text, clr):
        print(f"{clr}{text}{Style.RESET_ALL}")
    
    def Load_Weight_Simulation(self, _path=''):
           
        if self.mode == "PytorchSim_LN":  model = self.TorchSimulation_LN
        if self.mode == "PytorchSim_BN":  model = self.TorchSimulation_BN   
        if self.mode == "PythonSim"    :  model = self.PythonSimulation         

        self.pretrained_checkpoint = torch.load(_path,map_location='cpu')
        loaded_model = torch.load(_path,map_location='cpu')['model']

    
        
        if "scratch.pth" in _path:
            self.Show_Text(f'--> Starting training from scratch.'   , clr=Fore.BLUE)
            self.args.output_dir = self.args.output_dir + '/Scratch'

        else:   
            self.args.output_dir = self.args.output_dir + '/Pretrained'
            self.Show_Text(f"Pretrained Path            : {Fore.LIGHTYELLOW_EX}{_path}"   , clr=Fore.BLUE)
            

            if 'epoch' in self.pretrained_checkpoint.keys(): 
                self.Show_Text(f"Pretrained Epochs          : {Fore.LIGHTYELLOW_EX}{self.pretrained_checkpoint['epoch']}"   , clr=Fore.BLUE)    
        
        try:
            self.custom_model.load_state_dict(loaded_model)
            _model_state_dict = self.custom_model.state_dict()        
        except:
            _model_state_dict = loaded_model

        if "scratch.pth" in _path:   
            def rename_keys(old_dict, key_mapping):
                new_dict = {}
                for old_key, value in old_dict.items():
                    new_key = key_mapping.get(old_key, old_key)  # Use the new key if it exists, otherwise keep the old key
                    new_dict[new_key] = value
                return new_dict
            
            key_mapping = {
                'conv1.weight'  :'W0',
                'conv2.weight'  :'W1',
                'conv3.weight'  :'W2',
                'conv4.weight'  :'W3',
                'conv5.weight'  :'W4',
                'conv6.weight'  :'W5',
                'conv7.weight'  :'W6',
                'conv8.weight'  :'W7',
                'conv9.0.weight':'W8',
                'conv9.0.bias'  :'b8',
                'bn1.weight': 'gamma0',
                'bn2.weight': 'gamma1',
                'bn3.weight': 'gamma2',
                'bn4.weight': 'gamma3',
                'bn5.weight': 'gamma4',
                'bn6.weight': 'gamma5',
                'bn7.weight': 'gamma6',
                'bn8.weight': 'gamma7',
                'bn1.bias':'beta0',
                'bn2.bias':'beta1',
                'bn3.bias':'beta2',
                'bn4.bias':'beta3',
                'bn5.bias':'beta4',
                'bn6.bias':'beta5',
                'bn7.bias':'beta6',
                'bn8.bias':'beta7',
                'bn1.running_mean':'running_mean0',
                'bn2.running_mean':'running_mean1',
                'bn3.running_mean':'running_mean2',
                'bn4.running_mean':'running_mean3',
                'bn5.running_mean':'running_mean4',
                'bn6.running_mean':'running_mean5',
                'bn7.running_mean':'running_mean6',
                'bn8.running_mean':'running_mean7',
                'bn1.running_var':'running_var0',
                'bn2.running_var':'running_var1',
                'bn3.running_var':'running_var2',
                'bn4.running_var':'running_var3',
                'bn5.running_var':'running_var4',
                'bn6.running_var':'running_var5',
                'bn7.running_var':'running_var6',
                'bn8.running_var':'running_var7'
            }
            
            loaded_model = rename_keys(loaded_model, key_mapping)
        
        Weight = [loaded_model['W0'], loaded_model['W1'], loaded_model['W2'], loaded_model['W3'], loaded_model['W4'], loaded_model['W5'], 
                  loaded_model['W6'], loaded_model['W7'], loaded_model['W8']]
        Bias = loaded_model['b8']
        Gamma = [loaded_model['gamma0'], loaded_model['gamma1'], loaded_model['gamma2'], loaded_model['gamma3'], loaded_model['gamma4'],
                loaded_model['gamma5'], loaded_model['gamma6'], loaded_model['gamma7']]
        Beta = [loaded_model['beta0'], loaded_model['beta1'], loaded_model['beta2'], loaded_model['beta3'], loaded_model['beta4'], 
                loaded_model['beta5'], loaded_model['beta6'], loaded_model['beta7']]
        Running_Mean = [loaded_model['running_mean0'], loaded_model['running_mean1'], loaded_model['running_mean2'], 
                        loaded_model['running_mean3'], loaded_model['running_mean4'], loaded_model['running_mean5'], 
                        loaded_model['running_mean6'], loaded_model['running_mean7']]
        Running_Var = [loaded_model['running_var0'], loaded_model['running_var1'], loaded_model['running_var2'], loaded_model['running_var3'],
                       loaded_model['running_var4'], loaded_model['running_var5'], loaded_model['running_var6'], loaded_model['running_var7']]
    
        Outputs = Weight, Bias, Gamma, Beta, Running_Mean, Running_Var
        
        return Outputs

if __name__ == "__main__":
    # mode: [PytorchSim_BN, PytorchSim_LN, PythonSim]
    Mode = "Validation"
    YOLOv2Tiny_Simulation = YOLOv2Tiny_Simulation(Type="PytorchSim_BN")
    # Training
    if Mode == "Training":
        YOLOv2Tiny_Simulation.Run_Train()
    # Validation
    elif Mode == "Validation":
        YOLOv2Tiny_Simulation.Run_Validation()
    # Inference
    else: 
        YOLOv2Tiny_Simulation.Run_Infer()

