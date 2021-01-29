import numpy as np
import os
import torch
import pandas as pd
from shutil import copyfile
data_root_url = "../../egs/dataset/maestro"

data_train_url = "../../egs/dataset/maestro/train"
data_vaild_url = "../../egs/dataset/maestro/vaild"
data_test_url = "../../egs/dataset/maestro/test"

def makedir(data_url):
    if not os.path.exists(data_url):
        os.makedirs(data_url)

makedir(data_root_url)
makedir(data_train_url)
makedir(data_vaild_url)
makedir(data_test_url)

def get_subset(split_midi,data_type):
    return split_midi[split_midi[:,0]==data_type,1]
    #print(split_midi.split == data_type)
    #return split_midi[split_midi.split == data_type,'midi_filename'].values
source_root_url = '../../egs/dataset/maestro-v3.0.0/'
df = pd.DataFrame(pd.read_csv(source_root_url+'maestro-v3.0.0.csv'))
#print(df)
split_midi = df[['split','midi_filename']].values
#print(split_midi)
train_set = get_subset(split_midi,'train')
#print(train_set)
validation_set = get_subset(split_midi,'validation')

test_set = get_subset(split_midi,'test')

#copyfile(source_file, destination_file)

def transition(dataset,des_url):
    for item in dataset:
        copyfile(source_root_url+item, des_url+'/'+item[4:])

transition(train_set,data_train_url)
transition(validation_set,data_vaild_url)
transition(test_set,data_test_url)


