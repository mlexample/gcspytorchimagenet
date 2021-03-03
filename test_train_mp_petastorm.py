import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics as met
import torch_xla
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from petastorm.pytorch import DataLoader
from petastorm import make_reader, TransformSpec
import numpy as np
import sys
import os


for extra in ('/usr/share/torch-xla-1.7/pytorch/xla/test', '/pytorch/xla/test'):
    if os.path.exists(extra):
        sys.path.insert(0, extra)

import schedulers
# import gcsdataset
import args_parse

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--dataset': {
        'choices': ['gcsdataset', 'torchdataset'],
        'default': 'gcsdataset',
        'type': str,
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    opts=MODEL_OPTS.items(),
)


DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            DEFAULT_KWARGS, **{
                'lr': 0.5,
                'lr_scheduler_divide_every_n_epochs': 20,
                'lr_scheduler_divisor': 5,
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
    if getattr(FLAGS, arg) is None:
        setattr(FLAGS, arg, value)


def get_model_property(key):
    default_model_property = {
        'img_dim': 224,
        'model_fn': getattr(torchvision.models, FLAGS.model)
    }
    model_properties = {
        'inception_v3': {
            'img_dim': 299,
            'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
        },
    }
    model_fn = model_properties.get(FLAGS.model, default_model_property)[key]
    return model_fn

def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)

# train_size = spark.read.parquet(os.path.join(FLAGS.datadir, 'train/train')).count()
# test_size = spark.read.parquet(os.path.join(FLAGS.datadir, 'val/val')).count()
train_size = 1200000  # Roughly the size of Imagenet dataset.
test_size = 50000

img_dim = get_model_property('img_dim')

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.RandomResizedCrop(img_dim),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])
  
def _transform_row(parq_row):
    result_row = {
        'image': transform(parq_row['image']),
        'noun_id': parq_row['noun_id']
        }
    return result_row

transform = TransformSpec(_transform_row, removed_fields=['text'])

# def train_dataloader():
#     reader = make_reader(dataset_url=os.path.join(FLAGS.datadir, 'train/train'), reader_pool_type='thread', workers_count=10, 
#                          results_queue_size=256, 
#                          shuffle_row_groups=True, shuffle_row_drop_partitions=1,
#                          num_epochs=None,
#                          cur_shard=xm.get_ordinal(), shard_count=xm.xrt_world_size(),
#                          transform_spec=transform)
#     dataloader = DataLoader(reader, batch_size=FLAGS.batch_size, shuffling_queue_capacity=4096)
    
#     return dataloader

# def test_dataloader():
#     reader = make_reader(dataset_url=os.path.join(FLAGS.datadir, 'val/val'), 
#                          num_epochs=1, shuffle_row_groups=False, transform_spec=transform)
#     dataloader = DataLoader(reader, batch_size=FLAGS.test_set_batch_size)

#     return dataloader

def train_imagenet():
    print('==> Preparing data..')
    # img_dim = get_model_property('img_dim')
    # train_loader = train_dataloader()
    # test_loader = test_dataloader()
    train_loader = DataLoader(make_reader(dataset_url=os.path.join(FLAGS.datadir, 'train/train'), reader_pool_type='thread', workers_count=10,
                                          results_queue_size=256,
                                          shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                                          num_epochs=None,
                                          cur_shard=xm.get_ordinal(), shard_count=xm.xrt_world_size(),
                                          transform_spec=transform),
                                          batch_size=FLAGS.batch_size, shuffling_queue_capacity=4096)

    test_loader = DataLoader(make_reader(dataset_url=os.path.join(FLAGS.datadir, 'val/val'),
                                         num_epochs=1, shuffle_row_groups=False, transform_spec=transform),
                                         batch_size=FLAGS.test_set_batch_size)
    
    torch.manual_seed(42)
    
    device = xm.xla_device()
    model = get_model_property('model_fn')().to(device)
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=1e-4)
    num_training_steps_per_epoch = train_size // ( ### STEPS per EPOCH
        FLAGS.batch_size * xm.xrt_world_size())
    lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
        optimizer,
        scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
        scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
        scheduler_divide_every_n_epochs=getattr(
            FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
        num_steps_per_epoch=num_training_steps_per_epoch,
        summary_writer=writer)
    loss_fn = nn.CrossEntropyLoss()

    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        iters = 0
        for batch_idx, row in enumerate(loader):
            iters += 1
            if batch_idx % 1000 == 0:
                print("Step {}".format(batch_idx))
            # data, target = row['image'].to(device), row['noun_id'].to(device)
            data, target = row['image'], row['noun_id']
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)
            if lr_scheduler:
                lr_scheduler.step()
            if batch_idx % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, batch_idx, loss, tracker, epoch, writer))
          
    def test_loop_fn(loader, epoch):
        total_samples, correct = 0, 0
        model.eval()
        for batch_idx, row in enumerate(loader):
            # data, target = row['image'].to(device), row['noun_id'].to(device)
            data, target = row['image'], row['noun_id']
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]
            if batch_idx % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    test_utils.print_test_update, args=(device, None, epoch, batch_idx))
        correct_val = correct.item()
        accuracy_replica = 100.0 * correct_val / total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy_replica, np.mean)
        return accuracy, accuracy_replica

    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(
            epoch, test_utils.now()))
        train_loop_fn(train_device_loader, epoch)
        xm.master_print('Epoch {} train end {}'.format(
            epoch, test_utils.now()))
        accuracy, accuracy_replica = test_loop_fn(test_device_loader, epoch)
        xm.master_print('Epoch {} test end {}, Reduced Accuracy={:.2f}, Replica Accuracy={:.2f}'.format(
            epoch, test_utils.now(), accuracy, accuracy_replica))
        max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            dict_to_write={'Accuracy/test': accuracy},
            write_xla_metrics=True)
        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())
    test_utils.close_summary_writer(writer)
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return max_accuracy

def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    accuracy = train_imagenet()
    if accuracy < FLAGS.target_accuracy:
        print('Accuracy {} is below target {}'.format(accuracy,FLAGS.target_accuracy))
    sys.exit(21)

if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)