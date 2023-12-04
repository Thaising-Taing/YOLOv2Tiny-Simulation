import pickle
import sys
sys.path.append("../")

_Dataset = False
if _Dataset:
    from Dataset.factory import get_imdb
    from Dataset.roidb import RoiDataset, detection_collate

    dataset = 'voc0712trainval'
    imdb_name = 'voc_2007_trainval+voc_2012_trainval'
    imdbval_name = 'voc_2007_test'


    def axpy(N: int = 0., ALPHA: float = 1., X: int = 0, INCX: int = 1, Y: float = 0., INCY: int = 1):
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

    im_data = im_data[0].unsqueeze(0)
    gt_boxes = gt_boxes[0].unsqueeze(0)
    gt_classes = gt_classes[0].unsqueeze(0)
    num_boxes = num_boxes[0].unsqueeze(0)

    __data = im_data, gt_boxes, gt_classes, num_boxes

    # Path("Input").mkdir(parents=True, exist_ok=True)
    with open('/home/msis/Desktop/pcie_python/GUI/Post_Processing_Scratch/Input_Data_Batch8.pickle', 'wb') as handle:
        pickle.dump(__data, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    default_data = "/home/msis/Desktop/pcie_python/GUI/Post_Processing_Scratch/Input_Data_Batch8.pickle"
    with open(default_data, 'rb') as handle:
        b = pickle.load(handle)
    im_data, gt_boxes, gt_classes, num_boxes = b
    __data = im_data, gt_boxes, gt_classes, num_boxes
    # print(f"\n\nLoading data from saved file\n\nImage (im_data[0,:3,66:69,66:69]\n{im_data[0, :3, 66:69, 66:69]}\n\n")
