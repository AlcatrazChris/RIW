#Hyperparamenters
clamp = 2.0
batch_size = 2
val_batch_size = 2
test_batch_size = 2
init_scale = 1.0
lr = 1e-9
weight_decay = 1e-4
betas = (0.5, 0.999)
epoch = 1000

channels_in = 3

lamda_guide = 3
lamda_reconstruction = 5
lamda_low_frequency = 2
# lamda_color_l = 5

#train
cropsize = 256
weight_step = 50
gamma = 0.1
device_ids = [0]

#val
cropsize_val = 1024
val_freq = 5

#path
train_folder = 'dataset/train/'
val_folder = 'dataset/val/'
test_folder = 'dataset/test/'
MODEL_PATH = 'model/'
METRIC_PATH = 'runs/metrics'
IMAGE_PATH = 'runs/test'