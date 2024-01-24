# anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
anchors = [[1.08,1.19],  [3.42,4.41],  [6.63,11.38],  [9.42,5.11],  [16.62,10.52]]

object_scale = 5
noobject_scale = 1
class_scale = 1
coord_scale = 1

saturation = 1.5
exposure = 1.5
hue = .1

jitter = 0.2

thresh = .6

batch_size = 64
subdivision = 2

# lr = 0.0001
lr = 0.001

decay_lrs = {
    #  1: 0.1,
    # 10: 0.01,
    1  : 0.001,
    10 : 0.0001,
    50 : 0.00001,
    150: 0.000001
}

momentum = 0.9
weight_decay = 0.0005


# multi-scale training:
# {k: epoch, v: scale range}
multi_scale = True

# number of steps to change input size
scale_step = 40

scale_range = (3, 4)

epoch_scale = {
    1:  (3, 4),
    15: (2, 5),
    30: (1, 6),
    60: (0, 7),
    75: (0, 9)
}

input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)]

input_size = (416, 416)

test_input_size = (416, 416)

strides = 32

debug = False

