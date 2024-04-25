#Hyperparamenters
clamp = 2.0
batch_size = 2
val_batch_size = 4
test_batch_size = 2
init_scale = 0.9
lr = 5e-8
weight_decay = 1e-4
betas = (0.5, 0.999)
epoch = 1000

channels_in = 3

lamda_guide = 1
lamda_reconstruction = 7
lamda_low_frequency = 1
lamda_color_l = 5

#train
cropsize = 224
weight_step = 100
gamma = 0.9
device_ids = [0]

#val
cropsize_val = 1024
val_freq = 5

#path
train_folder = 'dataset/train/'
val_folder = 'dataset/val/'
test_folder = 'dataset/test/'
MODEL_PATH = 'model/'
METRIC_PATH = 'result/metrics'
IMAGE_PATH = 'result/images'