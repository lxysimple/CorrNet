import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
from torch.cuda.amp import autocast as autocast

class Processor():
    def __init__(self, arg):
        self.arg = arg
        #创建arg.work_dir，如果存在可选择是否覆盖
        if os.path.exists(self.arg.work_dir):
            answer = input('Current dir exists, do you want to remove and refresh it?\n')
            if answer in ['yes','y','ok','1']:
                print('Dir removed !')
                shutil.rmtree(self.arg.work_dir)
                os.makedirs(self.arg.work_dir)
            else:
                print('Dir Not removed !')
        else:
            os.makedirs(self.arg.work_dir)

        #将当前文件、baseline、tconv、resnet复制到arg.work_dir中
        # 为了防止出现意外，改了文件，先备份一下    
        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
        shutil.copy2('./modules/tconv.py', self.arg.work_dir)
        shutil.copy2('./modules/resnet.py', self.arg.work_dir)

        #创建一个Recorder对象（self.recoder）用于记录训练的状态和性能指标，这个对象是自己定义的
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        #将当前训练的配置参数（self.arg）保存到arg.work_dir中  
        self.save_arg()

        #创建一个随机种子，保证实验的可重复性，这个过程也是自己定义的
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        #用于多GPU训练，这个api是自己写的
        self.device = utils.GpuDataParallel()
        #再次记录训练状态和性能指标
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        #定义数据集、数据加载器、词表(来源于文件)、模型输出类别数
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        #载入模型和优化器，这个加载过程也是自己定义的
        self.model, self.optimizer = self.loading()

    #定义训练、测试过程
    def start(self):
        if self.arg.phase == 'train':
            best_dev = 100.0
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            #每个epoch
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder)
                if eval_model:
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                if dev_wer < best_dev:
                    best_dev = dev_wer
                    best_epoch = epoch
                    model_path = "{}_best_model.pt".format(self.arg.work_dir)
                    self.save_model(epoch, model_path)
                    self.recoder.print_log('Save best model')
                self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))
                if save_model:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
                epoch_time = time.time() - epoch_time
                total_time += epoch_time
                torch.cuda.empty_cache()
                self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
            self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                print('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
                               "dev", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
                                "test", 6667, self.arg.work_dir, self.recoder, self.arg.evaluate_tool)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        self.kernel_sizes = model.conv1d.kernel_size
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            raise ValueError("AMP equipped with DataParallel has to manually write autocast() for each forward function, you can choose to do this by yourself")
            #model.conv2d = nn.DataParallel(model.conv2d, device_ids=self.device.gpu_list, output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recoder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")
    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed)+worker_id)
    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
            pin_memory=True,
            worker_init_fn=self.init_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


#解析命令行参数，导入配置，开始训练
if __name__ == '__main__':
    #返回一个参数解析器
    #参数解释器就是当命令行输入如 --device 0，解释器就会解析并执行该命令
    #可以点进去看一看解析所执行的代码
    sparser = utils.get_parser()
    #对命令行输入的参数进行解析，将解析结果赋值给p变量
    p = sparser.parse_args()
    #sparser中默认配置文件在configs/baseline_iter.yaml
    # p.config = "baseline_iter.yaml"
    #如果p中config参数有指定，那么打开相应的配置文件，并读取其中的配置信息
    #否则就加载configs/baseline_iter.yaml配置文件
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                #使用yaml模块解析读取的配置信息，将其赋值给default_arg变量
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        #指定脚本中是否有不该存在的参数，有，报错终止
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        #将读取到的配置，设置为解析器默认值
        sparser.set_defaults(**default_arg)
    #解析器加载配置，将结果返回args
    args = sparser.parse_args()
    #打开args中的一个配置信息文件，用yaml解析并将结果传给args.dataset_info
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    
    #ai 初始化的一些工作
    processor = Processor(args)
    #备份一份
    utils.pack_code("./", args.work_dir)
    #开始训练
    processor.start()
