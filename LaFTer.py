import argparse
import torch
import pandas as pd
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
# custom
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.food101
import datasets.sun397
import datasets.ucf101
import datasets.imagenet_r
import datasets.imagenet
import datasets.imagenet_s
import datasets.imagenet_a
import datasets.caltech101
import datasets.cifar
import trainers.LaFTer as lafter_uft
from utils.utils import *
import os
from dassl.utils import Registry
from datasets import RESISC45


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


def test(args, teloader, model):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pl = AverageMeter('Acc@1', ':6.2f')
    one_hot = []
    one_hot_pl = []

    for i, (inputs) in enumerate(tqdm(teloader)):
        img = inputs["img"]
        labels = inputs["label"]

        if args.zero_shot:
            with torch.no_grad():
                output_pseudo_label = model(inputs.cuda(), zero_shot=True)
                _, predicted_pl = output_pseudo_label.max(1)
                one_hot_pl.append(predicted_pl.eq(labels.cuda()).cpu())
                acc1_pl = one_hot_pl[-1].sum().item() / len(labels)
                top1_pl.update(acc1_pl, len(labels))

        else:
            with torch.no_grad():
                inputs, labels = img.cuda(), labels.cuda()
                outputs = model(inputs, clip_eval=True)
                _, predicted = outputs.max(1)
                one_hot.append(predicted.eq(labels).cpu())
                acc1 = one_hot[-1].sum().item() / len(labels)
                top1.update(acc1, len(labels))

    if not args.zero_shot:
        return top1.avg * 100, top1_pl.avg * 100
    else:
        return top1_pl.avg * 100


def train_txt_cls(args, model):
    optimizer, _, _ = setup_text_training_utils(args, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in tqdm(range(args.txt_epochs)):
        loss = model.train_txt_clas(criteria)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.txt_cls_init()
    

def train_lafter(args, model, tr_loader, val_loader, test_loader=None):

    # first train text classifier
    train_txt_cls(args, model)
    

    all_acc = list()
    optimizer, scheduler, criteria = setup_lafter_training_utils(args, model)
    if args.svl_pl:
        #  initialize the svl adapter
        model.svl_adapter_init(args=args)
    if args.ln_frozen:
        print("------LN Frozen------")
        #Freeze CLIP
        for param in model.model.parameters():
            param.requires_grad = False
    
    # Print learnable parameters
    print('<<<<<<<<<<<<<<<<<<<<<<Learnable Parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(name)

    batch_time = lossmeter()
    data_time = lossmeter()
    best_acc = 0
    columns = ['Epoch', 'PS Text Acc','PS ZS Acc', 'Epoch Loss', 'Validation Accuracy', 'Test Accuracy','Best Model']
    # df = pd.DataFrame(columns=columns)
    df_to_append = []
    early_stopping_counter = 0
    early_stopping_threshold = 30 

    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.adapter.train()
        end = time.time()
        

        pl_text_acc = lossmeter()
        pl_zs_acc = lossmeter()
        total_loss = lossmeter()
        pl_svl_acc = lossmeter()

        print("-------------------------------------")
        print(args.bws)
        print("-------------------------------------")

        for i, batch in enumerate((tr_loader)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(model.device)

            optimizer.zero_grad()

            with torch.no_grad():
                output_text = model.forward_normal_for_pl(input[0])
            out = model.forward_aug_with_prompts(input[1].float().cuda())

            pseudo_label_text = F.softmax(output_text, dim=-1)  # / 0.04
            pseudo_label_text = pseudo_label_text.argmax(dim=1, keepdim=True)
            pseudo_label_text = pseudo_label_text.flatten().cuda()
            pl_text_acc.update((pseudo_label_text == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

            if not args.text_only:
                # Get Pseudo Label from Zero-Shot
                with torch.no_grad():
                    output_zs = model.forward_pl_zeroshot(input[0])
                    pseudo_label_zero_shot = F.softmax(output_zs, dim=-1).argmax(dim=1, keepdim=True)
                    pseudo_label_zero_shot = pseudo_label_zero_shot.flatten().cuda()
                    pl_zs_acc.update((pseudo_label_zero_shot == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

                # clip_conf = output_zs.softmax(dim=-1).max(dim=-1).values.mean().item

                if "fixed_alpha" in args.bws:
                    # Choose a value for alpha in the range [0, 1]
                    alpha = args.bws.split('_')[-1]
                    combined_tensor = alpha * output_zs + (1 - alpha) * output_text
                    average_tensor = torch.mean(combined_tensor, dim=1)
                    pseudo_label_text = F.softmax(average_tensor, dim=0)
                    pseudo_label_text = pseudo_label_text.argmax()
                    if pseudo_label_text.dim() > 0:
                        pseudo_label_text = pseudo_label_text.view(-1)

                    # #Fixed Alpha
                    # alpha = 0.2
                    # # Compute the new pl based on Alpha: pl_new = alpha*out_zero_shot + (1-alpha)*out_text
                    # pl_new = (output_zs*alpha +  output_text*(1-alpha))
                    # pl_new = torch.flatten(F.softmax(pl_new, dim=-1).argmax(dim=1, keepdim=True))
                    # pseudo_label_text = pl_new

                elif args.bws=="avg":
                    # Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
                    combined_tensor = torch.stack([output_zs, output_text], dim=2)
                    # Average along the new dimension
                    average_tensor = torch.mean(combined_tensor, dim=2)
                    pseudo_label_text = F.softmax(average_tensor, dim=1)
                    pseudo_label_text = pseudo_label_text.argmax(dim=1)
                    # Ensure pseudo_label_text is 1D or flatten it
                    if pseudo_label_text.dim() > 1:
                        pseudo_label_text = pseudo_label_text.view(-1)

                elif args.bws=="conf_alpha":
                    # BWS Computation: Alpha = softmax(concat(pl_zs,pl_text))
                    alpha = torch.cat([torch.max(F.softmax(output_zs),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_text),dim=1)[0].unsqueeze(1)], dim=-1)
                    alpha = F.softmax(alpha, dim=-1)
                    # New Psuedo Label
                    pl_new = (output_zs*alpha[:, 0].unsqueeze(1) +  output_text*alpha[:, 1].unsqueeze(1))
                    pl_new = torch.flatten(F.softmax(pl_new, dim=-1).argmax(dim=1, keepdim=True))
                    #Change later
                    pseudo_label_text = pl_new

            if args.svl_pl:

                # Get Pseudo Label from SVL
                # with torch.no_grad():
                output_svl = model.forward_svl(input[0])
                pseudo_label_svl = F.softmax(output_svl, dim=-1).argmax(dim=1, keepdim=True)
                pseudo_label_svl = pseudo_label_svl.flatten().cuda()
                pl_svl_acc.update((pseudo_label_svl == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

                with torch.no_grad():
                    output_zs = model.forward_pl_zeroshot(input[0])
                    pseudo_label_zero_shot = F.softmax(output_zs, dim=-1).argmax(dim=1, keepdim=True)
                    pseudo_label_zero_shot = pseudo_label_zero_shot.flatten().cuda()
                    pl_zs_acc.update((pseudo_label_zero_shot == batch["label"].cuda()).sum().item() / len(batch["label"]), len(batch["label"]))

                if args.bws=="avg":
                    # Combine the tensors along a new dimension (e.g., concatenate along a new dimension)
                    combined_tensor = torch.stack([output_zs, output_text, output_svl], dim=2)
                    # Average along the new dimension
                    average_tensor = torch.mean(combined_tensor, dim=2)
                    pseudo_label_text = F.softmax(average_tensor, dim=1)
                    pseudo_label_text = pseudo_label_text.argmax(dim=1)
                    # Ensure pseudo_label_text is 1D or flatten it
                    if pseudo_label_text.dim() > 1:
                        pseudo_label_text = pseudo_label_text.view(-1)

                elif args.bws=="conf_alpha":
                    # BWS Computation: Alpha = softmax(concat(pl_zs,pl_text))
                    alpha = torch.cat([torch.max(F.softmax(output_zs),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_text),dim=1)[0].unsqueeze(1),torch.max(F.softmax(output_svl),dim=1)[0].unsqueeze(1)], dim=-1)
                    alpha = F.softmax(alpha, dim=-1)
                    # New Psuedo Label
                    pl_new = (output_zs*alpha[:, 0].unsqueeze(1) +  output_text*alpha[:, 1].unsqueeze(1) + output_svl*alpha[:, 2].unsqueeze(1))
                    pl_new = torch.flatten(F.softmax(pl_new, dim=-1).argmax(dim=1, keepdim=True))
                    #Change later
                    pseudo_label_text = pl_new 

            loss = criteria(out.squeeze(), pseudo_label_text)
            total_loss.update(loss.item(),len(tr_loader))

            if i % args.print_freq == 0:
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "loss {losses}\t"
                    "lr {lr:.6e}".format(
                        epoch,
                        args.epochs,
                        i + 1,
                        len(tr_loader),
                        losses=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    ))

            loss.backward()
            optimizer.step()
        scheduler.step()
        
        
        print(f'Epoch Loss: {total_loss.avg}')

        print(f'Evaluation: {epoch}')
        val_acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {val_acc}')
        all_acc.append(val_acc)

        ps_text_acc=pl_text_acc.avg
        ps_zs_acc=pl_zs_acc.avg


        print(f'Pseudo Label Text Accuracy: {pl_text_acc.avg}')
        print(f'Pseudo Label Zero Shot Accuracy: {pl_zs_acc.avg}')
        print(f'Pseudo Label SVL Accuracy: {pl_svl_acc.avg}')

        best_test_acc=None
        if val_acc>best_acc:
            best_val_acc="Yes"
            best_acc=val_acc
            early_stopping_counter = 0 
            print('------------')
            print("Best Epoch ", epoch)
            print("Best Val acc", val_acc)
            best_test_acc = test_prompting(test_loader, model)
            print("Test acc ", best_test_acc)
            print('------------')

            #Save the whole model
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pth")) 
        else:
            early_stopping_counter += 1
        df_to_append.append([epoch, ps_text_acc, ps_zs_acc, total_loss.avg, val_acc, best_test_acc, best_val_acc])
        print("Output dir: ",args.output_dir)
        if early_stopping_counter >= early_stopping_threshold:
            print(f'Early stopping at epoch {epoch} due to no improvement in validation accuracy.')
            break
    df = pd.DataFrame(df_to_append, columns=columns)    
    csv_path = os.path.join(args.output_dir, "training_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f'-------------------------------- Best Validation Accuracy: {max(all_acc)} --------------------------------')
    print(f'-------------------------------- Best Validation Accuracy Epoch: {all_acc.index(max(all_acc))} --------------------------------')
    
def main(args):
    cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    dataset_registary = Registry("Dataset")

    dataset_registary.register(RESISC45)

    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args
    test_loader = trainer.test_loader
    val_loader = trainer.val_loader
    train_loader = trainer.train_loader_x

    if args.zero_shot:
        zero_shot(model, test_loader)
        # acc = test_prompting(test_loader, model, model_path="/home/mohamed.imam/Thesis/RS_zero_shot/output/LaFTer/vit_b32/resisc45/model_best.pth")
        # print(f'final accuracy:{acc}')
    else:
        train_lafter(args, model,train_loader, val_loader, test_loader=test_loader)
        test_acc = test_prompting(test_loader, model, model_path=os.path.join(args.output_dir,"model_best.pth"))
        print(f'Test accuracy (loading saved model):{test_acc}')
        # val_acc = test_prompting(val_loader, model, model_path=os.path.join(args.output_dir,"model_best.pth"))
        # print(f'Val accuracy (loading saved model):{val_acc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=7777, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='tap', required=True, choices=['cls_only',
                                                                                      'templates_only', 'lafter', 'zero_shot'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--logfolder', default='logs', type=str)
    parser.add_argument('--text_only', action="store_true")
    parser.add_argument('--bws', type=str, default="None", choices=['conf_alpha','fixed_alpha_0.25', 'avg'])
    parser.add_argument('--ln_frozen', action="store_true")
    parser.add_argument('--svl_pl', action="store_true")
    parser.add_argument('--svl_model_path', type=str, default=None)
    args = parser.parse_args()
    args.mile_stones = None
    
    main(args)

