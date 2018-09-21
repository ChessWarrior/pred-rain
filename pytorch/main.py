from utils.imports import *
from distributed import DistributedDataParallel as DDP
import models
from dataloader import torch_loader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch PredRain')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--save-dir', type=str, default=Path.home()/'imagenet_training',
                    help='Directory to save logs and models.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cycle-len', default=1, type=float, metavar='N',
                    help='Length of cycle to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--use-tta', action='store_true', help='Validate model with TTA at the end of traiing.')
parser.add_argument('--train-128', action='store_true', help='Train model on 128. TODO: allow custom epochs and LR')
parser.add_argument('--sz',       default=224, type=int, help='Size of transformed image.')
# parser.add_argument('--decay-int', default=30, type=int, help='Decay LR by 10 every decay-int epochs')
parser.add_argument('--use-clr', type=str, 
                    help='div,pct,max_mom,min_mom. Pass in a string delimited by commas. Ex: "20,2,0.95,0.85"')
parser.add_argument('--loss-scale', type=float, default=1,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--prof', dest='prof', action='store_true', help='Only run a few iters for profiling.')

parser.add_argument('--dist-url', default='file://sync.file', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

parser.add_argument('--world-size', default=1, type=int,
                    help='Number of GPUs to use. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')
parser.add_argument('--rank', default=0, type=int,
                    help='Used for multi-process training. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')




# This is important for speed
cudnn.benchmark = True
global args
# args = parser.parse_args()
# print('Running script with args:', args)


def main():
    args.distributed = args.world_size > 1
    args.gpu = 0
    if args.distributed: args.gpu = args.rank % torch.cuda.device_count()

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)

    if args.fp16: assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    if args.cycle_len > 1: args.cycle_len = int(args.cycle_len)

    # create model
    if args.pretrained: model = models.__dict__[args.arch](pretrained=True)
    else:               model = models.__dict__[args.arch]()

    model = model.cuda()
    if args.distributed: model = DDP(model)

    if args.train_128: data, train_sampler = torch_loader(f'{args.data}-160', 128)
    else:              data, train_sampler = torch_loader(args.data, args.sz)

    learner = Learner.from_model_data(model, data)
    learner.crit = F.cross_entropy
    learner.metrics = [accuracy, top5]
    if args.fp16: learner.half()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.resume.endswith('.h5'): args.resume = args.resume[:-len('.h5')]
            learner.load(args.resume)
        else: print("=> no checkpoint found at '{}'".format(args.resume))    

    if args.prof:
        args.epochs = 1
        args.cycle_len=1
    if args.use_clr: args.use_clr = tuple(map(float, args.use_clr.split(',')))

    # 128x128
    if args.train_128:
        save_dir = f'{args.save_dir}/128'
        update_model_dir(learner, save_dir)
        sargs = save_args('first_run_128', save_dir)
        learner.fit(args.lr,args.epochs, cycle_len=args.cycle_len,
                    sampler=train_sampler, wds=args.weight_decay, use_clr_beta=args.use_clr,
                    loss_scale=args.loss_scale, **sargs)
        save_sched(learner.sched, save_dir)
        data, train_sampler = torch_loader(args.data, args.sz)
        learner.set_data(data)


    # Full size
    update_model_dir(learner, args.save_dir)
    sargs = save_args('first_run', args.save_dir)
    learner.fit(args.lr,args.epochs, cycle_len=args.cycle_len,
                sampler=train_sampler, wds=args.weight_decay, use_clr_beta=args.use_clr,
                loss_scale=args.loss_scale, **sargs)
    save_sched(learner.sched, args.save_dir)

    # TTA works ~50% of the time. Hoping top5 works better
    if args.use_tta:
        log_preds,y = learner.TTA()
        preds = np.mean(np.exp(log_preds),0)
        acc = accuracy(torch.FloatTensor(preds),torch.LongTensor(y))
        t5 = top5(torch.FloatTensor(preds),torch.LongTensor(y))
        print('TTA acc:', acc)
        print('TTA top5:', t5[0])

        with open(f'{args.save_dir}/tta_accuracy.txt', "a", 1) as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S")+f"\tTTA accuracy: {acc}\tTop5: {t5}")

    print('Finished!')