import sys
sys.path.append('/data2/qt/MusicGeneration/mg/model/Musictransformer')
from network import MusicTransformer
from metrics import *
from criterion import SmoothCrossEntropyLoss, CustomSchedule
import config
from data import Data

import utils
import datetime
import time
import optparse
import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter


def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--save_path',
                      dest='save_path',
                      type='string',
                      default='/data2/qt/MusicGeneration/mg/model/MusicTransformer/save_model/')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='/data2/qt/MusicTransformer-pytorch/dataset/processed/')

    parser.add_option('-e', '--epochs',
                      dest='epochs',
                      type='int',
                      default=20000)

    parser.add_option('-i', '--saving-interval',
                      dest='saving_interval',
                      type='int',
                      default=50)

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=config.train['batch_size'])

    parser.add_option('-l', '--learning-rate',
                      dest='learning_rate',
                      type='float',
                      default=config.train['learning_rate'])

    parser.add_option('-w', '--window-size',
                      dest='window_size',
                      type='int',
                      default=config.train['window_size'])

    parser.add_option('-S', '--stride-size',
                      dest='stride_size',
                      type='int',
                      default=config.train['stride_size'])

    parser.add_option('-g', '--multi_gpu',
                      dest='multi_gpu',
                      type='string',
                      default='False')

    parser.add_option('-m', '--load_path',
                      dest='load_path',
                      type='string',
                      default=None)

    parser.add_option('-M', '--max_seq',
                      dest='max_seq',
                      type='int',
                      default=2048)

    return parser.parse_args()[0]

options = get_options()
# ------------------------------------------------------------------------

saving_interval = options.saving_interval
window_size = options.window_size
stride_size = options.stride_size

data_path = options.data_path
l_r = options.learning_rate
batch_size = options.batch_size
pickle_dir = options.data_path
max_seq = options.max_seq
epochs = options.epochs
load_path = options.load_path
save_path = options.save_path

if options.multi_gpu == 'True' :
    multi_gpu = True
else:
    multi_gpu = False


event_dim = config.event_dim #EventSeq.dim()
model_config = config.model

device = config.device
limlen = config.window_size

print('-' * 70)

print('Save path:', save_path)
print('Dataset path:', data_path)
print('Saving interval:', saving_interval)
print('-' * 70)

print('Hyperparameters:', utils.dict2params(model_config))
print('Learning rate:', l_r)
print('Batch size:', batch_size)
print('Window size:', window_size)
print('Stride size:', stride_size)
print('Device:', device)
print('-' * 70)


# ========================================================================
# Load model and dataset
# ========================================================================

def load_model():
    global model_config, device
    model = MusicTransformer(**model_config)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    model.to(device)
    return model

"""
def load_dataset(limlen):
    global data_path
    dataset = Event_Dataset(data_path, limlen, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset
"""

print('Loading model')
mt = load_model()
print(mt)

# print('-' * 70)
#
# print('Loading dataset')
# # print(os.path.isdir(data_path))
# dataset = load_dataset(limlen)
# print(dataset)

print('-' * 70)


# ------------------------------------------------------------------------

def save_model(epoch, acc = 0.0):
    global mt, save_path
    #torch.save(single_mt.state_dict(), args.model_dir+'/train-{}-{}.pth'.format(e, eval_metrics['accuracy']))
    print('Saving to', save_path+'train-{}-{}.pth'.format(epoch, acc))
    torch.save(mt.state_dict(),  save_path+'train-{}-{}.pth'.format(epoch, acc))
    # torch.save({'model_config': model_config,
    #             'model_state': model.state_dict(),
    #             'model_optimizer_state': optimizer.state_dict()}, save_path)
    print('Done saving')


# ========================================================================
# Training
# ========================================================================


# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


# load data
print(config.pickle_dir)
dataset = Data(config.pickle_dir, config.max_seq)
print(dataset)


# load model
learning_rate = l_r

# # define model
# mt = MusicTransformer(
#     embedding_dim=config.embedding_dim,
#     vocab_size=config.vocab_size,
#     num_layer=config.num_layers,
#     max_seq=config.max_seq,
#     dropout=config.dropout,
#     debug=config.debug, loader_path=config.load_path
# )
# mt.to(config.device)
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# multi-GPU set
if torch.cuda.device_count() > 1:
    single_mt = mt
    mt = torch.nn.DataParallel(mt, output_device=torch.cuda.device_count()-1)
else:
    single_mt = mt

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size)
})

print(mt)
print('| Summary - Device Info : {}'.format(torch.cuda.device))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# train_log_dir = 'logs/'+config.experiment+'/'+current_time+'/train'
# eval_log_dir = 'logs/'+config.experiment+'/'+current_time+'/eval'

# train_summary_writer = SummaryWriter(train_log_dir)
# eval_summary_writer = SummaryWriter(eval_log_dir)

# Train Start
scheduler.optimizer.zero_grad()
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    try:
        print(">>> [Epoch was updated]")
        for b in range(len(dataset.file_dict['train']) // config.batch_size):

            try:
                batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
                batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
                batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
            except IndexError:
                continue

            start_time = time.time()
            mt.train()
            sample = mt.forward(batch_x)
            metrics = metric_set(sample, batch_y)
            loss = metrics['loss'] / config.accum_grad
            loss.backward()

            if (b+1) % config.accum_grad == 0:
                scheduler.step()
                # train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
                # train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)
                # train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
                # train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)
                scheduler.optimizer.zero_grad()
            end_time = time.time()

            # if config.debug:
            #     print("[Loss]: {}".format(loss))

            torch.cuda.empty_cache()
            idx += 1

            # switch output device to: gpu-1 ~ gpu-n
            sw_start = time.time()
            if torch.cuda.device_count() > 1:
                mt.output_device = idx % (torch.cuda.device_count() -1) + 1
            sw_end = time.time()
            # if config.debug:
            #     print('output switch time: {}'.format(sw_end - sw_start) )

            # result_metrics = metric_set(sample, batch_y)
        # single_mt.eval()
        # eval_x, eval_y = dataset.slide_seq2seq_batch(2, config.max_seq, 'eval')
        # eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
        # eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)
        #
        # eval_preiction, weights = single_mt.forward(eval_x)
        #
        # eval_metrics = metric_set(eval_preiction, eval_y)
        #
        # ##### save_model(e, eval_metrics['accuracy'])

        # if b == 0:
        #     # train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
        #     # train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
        #     for i, weight in enumerate(weights):
        #         attn_log_name = "attn/layer-{}".format(i)
        #         # utils.attention_image_summary(
        #         #     attn_log_name, weight, step=idx, writer=eval_summary_writer)

        # eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
        # eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
        # eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=idx)

        print('\n====================================================')
        print('Epoch/Batch: {}/{}'.format(e, b))
        print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['accuracy']))
        # print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['accuracy']))

    except KeyboardInterrupt:
        save_model(e)
        break

save_model('final')
# eval_summary_writer.close()
# train_summary_writer.close()


