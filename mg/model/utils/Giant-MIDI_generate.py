import numpy as np
import os
import torch
import pandas as pd
from shutil import copyfile
source_root_url = "../../egs/dataset/Giant-MIDI"
data_root_url = "../../egs/dataset/giant_midis"
data_train_url = "../../egs/dataset/giant_midis/train"
data_vaild_url = "../../egs/dataset/giant_midis/vaild"
data_test_url = "../../egs/dataset/giant_midis/test"

def makedir(data_url):
    if not os.path.exists(data_url):
        os.makedirs(data_url)

makedir(data_root_url)
makedir(data_train_url)
makedir(data_vaild_url)
makedir(data_test_url)

dataset = os.listdir(source_root_url)
print(dataset)
#copyfile(source_file, destination_file)
portion = int(0.1*len(dataset))
train_set = dataset[:-2*portion]
validation_set = dataset[-2*portion:-portion]
test_set = dataset[-portion:]

def transition(dataset,des_url):
    for item in dataset:
        copyfile(source_root_url+'/'+item, des_url+'/'+item)

transition(train_set,data_train_url)
transition(validation_set,data_vaild_url)
transition(test_set,data_test_url)


