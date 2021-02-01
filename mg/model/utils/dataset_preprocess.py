import os
import sys
sys.path.append('/data2/qt/MusicGeneration/mg/model/')
from utils.preprocess import *

data_root_url = "../../../egs/dataset/maestro"
data_train_url = "../../../egs/dataset/maestro/train"
save_train_url = "../../../egs/dataset/maestro/train_processed"
data_vaild_url = "../../../egs/dataset/maestro/vaild"
save_vaild_url = "../../../egs/dataset/maestro/vaild_processed"
data_test_url = "../../../egs/dataset/maestro/test"
save_test_url = "../../../egs/dataset/maestro/test_processed"

def makedir(data_url):
    if not os.path.exists(data_url):
        os.makedirs(data_url)

makedir(data_root_url)
makedir(data_train_url)
makedir(save_train_url)
makedir(data_vaild_url)
makedir(save_vaild_url)
makedir(data_test_url)
makedir(save_test_url)

def preprocess_dataset(midi_dir,save_dir):
    utils.preprocess.preprocess_midi_files_under(
        midi_root=midi_dir,
        save_dir=save_dir,
        num_workers=10,
        type='event')

preprocess_dataset(data_vaild_url,save_vaild_url)
preprocess_dataset(data_test_url,save_test_url)
preprocess_dataset(data_train_url,save_train_url)


