import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")
        
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], 
                             args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)
    
    dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], 
                           args=args, phase='dev')
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, 
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler, 
                                pin_memory=args.pin_mem)
        
    test_data = S2T_Dataset(path=test_label_paths[args.dataset], 
                            args=args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = Uni_Sign(
                args=args
                )
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=True)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )

    #################### for dataloader
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(1, args.epochs)
    print_freq = 10
    target_dtype = next(model.parameters()).dtype

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()


    target_dtype = next(model.parameters()).dtype

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)
        
        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = next(model.parameters()).dtype  # örn. torch.float16

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(stack_out, 
                                                max_new_tokens=100, 
                                                num_beams = 4,
                        )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id
    
    pad_tensor = torch.ones(150-len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
    
    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+f'/{phase}_tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+f'/{phase}_tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    """
    output_dir =".zout/finetuning_stage3"
    ckpt_path ="./ckpts/wlasl_pose_only_islr.pth" #"./ckpts/wlasl_rgb_pose_islr.pth"

    args.batch_size=1
    args.gradient_accumulation_steps =1
    args.epochs=20
    #args.opt = AdamW
    args.lr = 3e-4
    args.output_dir = output_dir
    args.finetune = ckpt_path
    args.dataset = "WLASL"
    args.task= "ISLR"
    args.max_length = 32
    args.rgb_support = False
    """
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    main(args)