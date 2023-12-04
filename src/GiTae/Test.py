import torch
import pickle
import torch.nn.functional as F
from Detection.Detection import *

with open("/home/msis/Desktop/pcie_python/GUI_List_Iter/Output_data/output.pickle", mode="rb") as f:
    output = pickle.load(f)

output = torch.tensor(output, dtype=torch.float32).reshape(8, 125, 13, 13)

def reshape_output(out):
    # print('Calculating the loss and its gradients for python model.')
    out = torch.tensor(out, requires_grad=True)
    
    # Additional Condition: 
    out = out[0:(8*125*(13**2))]

    out = out.reshape(8, 125, 13, 13)
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

    # return output_data
    return output_variable


if __name__ == "__main__":
    out = reshape_output(output)
    print(out[0].shape)


    #  Read input image (from dataloader, get path and read image)
    dataset = 'voc07test'
    if dataset == 'voc07trainval':
        imdbval_name = 'voc_2007_trainval'
    elif dataset == 'voc07test':
        imdbval_name = 'voc_2007_test'
    else:
        raise NotImplementedError

    val_imdb = get_imdb(imdbval_name)

    val_dataset = RoiDataset(val_imdb, train=False)
    print("Inference Dataset: " + str(len(val_dataset)))
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=detection_collate, drop_last=True)
    # Take full validation dataset with shuffle True
    # Make the iteration only 1
    
    # Small Dataset
    small_val_dataset = torch.utils.data.Subset(val_dataset, range(0, 1000))
    # small_val_dataset = torch.utils.data.Subset(val_dataset, range(0, 1))
    print("Sub Inference Dataset: " + str(len(small_val_dataset)))
    small_val_dataloader = DataLoader(small_val_dataset, batch_size=8, shuffle=False)
    # small_val_dataloader = DataLoader(small_val_dataset, batch_size=1, shuffle=False)
    # small_val_dataloader = DataLoader(small_val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=detection_collate, drop_last=True)
    
    # Total Epochs
    Inference_Epochs = 1
    
    # Iteration_Per_Epoch
    iters_per_epoch = int(len(small_val_dataset) / 8)
    print("Inference Iterations: " + str(iters_per_epoch)+"\n")

    detection(yolo_outputs=out, epoch=1, val_imdb=val_imdb, small_val_dataloader=small_val_dataloader)