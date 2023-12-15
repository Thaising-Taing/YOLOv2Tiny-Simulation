from pathlib import Path
import pickle
import torch
from python_2iteration import *

python_model = DeepConvNet(input_dims=(3, 416, 416),
						num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
						max_pools=[0, 1, 2, 3, 4],
						weight_scale='kaiming',
						batchnorm=True,
						dtype=torch.float32, device='cpu')

_Load_Weights = True
if _Load_Weights:
	model = Yolov2()
	weightloader = WeightLoader()
	python_model = weightloader.load(python_model, model, '../../Dataset/Dataset/pretrained/yolov2-tiny-voc.weights')

_Dataset = False
if _Dataset:
	from dataset.factory import get_imdb
	from dataset.roidb import RoiDataset, detection_collate
	dataset = 'voc0712trainval'
	imdb_name = 'voc_2007_trainval+voc_2012_trainval'
	imdbval_name ='voc_2007_test'

	def axpy(N: int = 0., ALPHA: float = 1., X: int = 0, INCX: int = 1, Y: float =0., INCY: int = 1):
		for i in range(N):
			Y[i * INCY] += ALPHA * X[i * INCX]

	def get_dataset(datasetnames):
		names = datasetnames.split('+')
		dataset = RoiDataset(get_imdb(names[0]))
		print('load dataset {}'.format(names[0]))
		for name in names[1:]:
			tmp = RoiDataset(get_imdb(name))
			dataset += tmp
			print('load and add dataset {}'.format(name))
		return dataset

	train_dataset = get_dataset(imdb_name)

_Dataloader = _Dataset
if _Dataloader:
	from torch.utils.data import DataLoader
	train_dataloader = DataLoader(train_dataset, batch_size=64,
									shuffle=True, num_workers=2,
									collate_fn=detection_collate, drop_last=True)
	train_data_iter = iter(train_dataloader)

_Get_Next_Data = _Dataloader
if _Get_Next_Data:
	_data = next(train_data_iter)
	im_data, gt_boxes, gt_classes, num_boxes = _data

	im_data     = im_data[0].unsqueeze(0)
	gt_boxes    = gt_boxes[0].unsqueeze(0)
	gt_classes  = gt_classes[0].unsqueeze(0)
	num_boxes   = num_boxes[0].unsqueeze(0)

	__data = im_data, gt_boxes, gt_classes, num_boxes

	# Path("Input").mkdir(parents=True, exist_ok=True)
	with open('Input_Data.pickle','wb') as handle:
		pickle.dump(__data,handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
	default_data = './Input_Data_Batch8.pickle'
	# if not os.path.isfile(default_data):
	# 	print('Deafult data file does not exist. Downlaoding file now.')
	# 	url = 'https://drive.google.com/uc?id=1j1zyq0lRQ_BVqSS5zM2GHU3Q4c1qH0DP'
	# 	output = default_data
	#	import gdown
	# 	gdown.download(url, output, quiet=False)
	with open(default_data, 'rb') as handle:
		b = pickle.load(handle)
	im_data, gt_boxes, gt_classes, num_boxes = b
	__data = im_data, gt_boxes, gt_classes, num_boxes
	print(f"\n\nLoading data from saved file\n\nImage (im_data[0,:3,66:69,66:69]\n{im_data[0,:3,66:69,66:69]}\n\n")
	

if __name__ == '__main__':
# Control Forward and Backward Prop
	python_model.forward_prop 		  = True 	# Perform forward propagation or load saved file.
	python_model.cal_loss 			  = True 	# Perform loss calculation or load save file
	python_model.backward_prop 		  = True 	# Perform backward propagation or load saved file
   
#  Save outputs  
	python_model.save_pickle 		  = False 	# Save output in form of pickle file
  
	python_model.save_layer_output 	  = False 	# Save output for each layers (not module)
	python_model.save_module_output	  = False 	# Save output for each module
   
	python_model.save_in_dec_format	  = False 	# Save output for each module in decimal format
	python_model.save_in_hex_format   = False 	# Save output in hex format
   
#  FP 32 to FP 16  
	# python_model.convert_to_fp16 	  = True 	# True to convert from FP32 to FP16
  
	# python_model.convert_forward   	  = True 	# Convert the Forward Propagation
	# python_model.convert_loss_grad 	  = True 	# Convert the loss gradients
	# python_model.convert_backward  	  = True 	# Convert the backward propagation
 
	# python_model.convert_layer_InOut  = True 	# Convert only layer input output     (Layer= Layer0, Layer1, etc)  #InOut
	# python_model.convert_module_InOut = True 	# Convert each module's input output  (Module = Conv, BN, etc)
  
	# python_model.conversion_method    = 'Ali' 	# 'Sawera'
	# python_model.convert_to_IEEE16 	  = True 	# False means Convert from FP32 to BFP16
 
	Fout, Fcache, loss, loss_grad, BlDout, Bgrads = python_model.train(im_data, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_boxes)