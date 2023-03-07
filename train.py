import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as tud
from torchvision.models import resnet50, resnext101_32x8d
from imagenet_dataset import ImgNet_C_val_Dst, ImgNet_C_train_Dst
from tqdm import tqdm
import csv
import json
import random
from UEM import FE_Net


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse():
    parser = argparse.ArgumentParser(description="train gurie")
    parser.add_argument("--bs", type=int, default=512, help="batch_size")
    parser.add_argument("--gpuid", type=str, default="0,1,2,3", help="GPU id to train")
    parser.add_argument("--method", type=str, default="uc_50", help="name of method")
    parser.add_argument("--mode", type=str, default="train", help="train_or_val")
    parser.add_argument("--ckpt_path", type=str, default="/home/yangzhou/Code/self_feature_distillation/checkpoints/",
                        help="ckpt_path to load/val")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    return args


class new_clsnet(nn.Module):
    def __init__(self, model):
        super(new_clsnet, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.f_net = FE_Net(2048, btn=6)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(*list(model.children())[-1:])

    def forward(self, clean=None, distort=None):

        f_dis = self.resnet_layer(distort)
        f_rec, theta = self.f_net(f_dis)
        vec_rec = self.avgpool(f_rec)
        pred_rec = self.fc(vec_rec.view(vec_rec.size(0), -1))
        if clean is not None:
            f_clean = self.resnet_layer(clean)
            vec_clean = self.avgpool(f_clean)
            pred_clean = self.fc(vec_clean.view(vec_clean.size(0), -1))
            return f_clean, f_rec, theta, pred_clean, pred_rec
        else:
            return pred_rec


class uc_loss(nn.Module):
    def __init__(self):
        super(uc_loss, self).__init__()

    def forward(self, gt, mu, theta):
        return torch.mean((gt - mu) ** 2 / (1e-6 + 2 * torch.exp(theta)) + theta)


@torch.no_grad()
def val(classifier, args, best_or_last):
    result_dict = {}
    if best_or_last == "best":
        classifier = load_ckpt(classifier, args=args, best_or_last=best_or_last)
    ann_path = "/home/yangzhou/datasets/imagenet/meta/val.txt"
    clean_img_path = "/home/yangzhou/datasets/imagenet/ILSVRC2012_img_val/"
    img_path = "/home/yangzhou/datasets/imagenet_c/"
    re_acc = 0.0
    with open("result.csv", 'a') as fcsv:
        crp_name = ["method", "clean", "gaussian_noise", "shot_noise", "impulse_noise",
                    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                    "snow", "frost", "fog",
                    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise",
                    "spatter", "gaussian_blur", "saturate"]
        severity_level = ["1", "2", "3", "4", "5"]
        f_csv = csv.writer(fcsv)
        # f_csv.writerow(crp_name)
        result = [args.method]
        for crp in crp_name[1:]:
            avg_acc = 0
            for severity in severity_level:
                datasets = ImgNet_C_val_Dst(clean_img_path, img_path, ann_path, crp, severity)
                dataloader = tud.DataLoader(dataset=datasets,
                                            batch_size=512,
                                            num_workers=16,
                                            pin_memory=True,
                                            shuffle=False,
                                            drop_last=False
                                            )
                acc = 0
                with torch.no_grad():
                    for img, ann in tqdm(dataloader):
                        pred = classifier(distort=img.cuda())
                        hit = np.count_nonzero(np.argmax(pred.detach().cpu().numpy(), axis=1) == ann.numpy())
                        acc += hit
                print(crp, ", ", severity, ", Acc = ", acc / len(datasets))  # 72.648%  76.55%
                if crp == "clean":
                    avg_acc = acc / len(datasets)
                    break
                avg_acc += acc / len(datasets) / 5
            result_dict[str(crp)] = avg_acc
            result.append(1 - avg_acc)
            if crp == "gaussian_noise":
                re_acc = avg_acc
        f_csv.writerow(result)
    fcsv.close()
    print(result_dict)
    return re_acc


def load_ckpt(model, optimizer=None, lr_schedule=None, args=None, best_or_last=None):
    load_path = os.path.join(args.ckpt_path, args.method)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    if len(os.listdir(load_path)) == 0:
        return 0, 0, 0, model, optimizer, lr_schedule
    load_path = os.path.join(load_path, "%s.pth" % best_or_last)

    state_dict = torch.load(load_path)
    model.module.load_state_dict(state_dict["model"])
    if optimizer is None:
        return classifier
    optimizer.load_state_dict(state_dict["optimizer"])
    lr_schedule.load_state_dict(state_dict["lr_schedule"])
    epoch = state_dict["epoch"]
    itrs = state_dict["itrs"]
    acc = state_dict["acc"]
    # optimizer.state_dict()['param_groups'][0]['lr'] = 1e-4
    print("now lr", optimizer.state_dict()['param_groups'][0]['lr'])
    print("epoch: ", epoch)
    return epoch, itrs, acc, model, optimizer, lr_schedule


def save_ckpt(epoch, itrs, model, optimizer, lr_schedule, acc, args, is_best=True):
    save_path = os.path.join(args.ckpt_path, args.method)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    state_dict = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch,
        "itrs": itrs
    }
    if is_best:
        state_dict["best_acc"] = acc
        torch.save(state_dict, os.path.join(save_path, "best.pth"))
    else:
        state_dict["acc"] = acc
        torch.save(state_dict, os.path.join(save_path, "last.pth"))


def train(classifier, args):
    img_path = "/home/yangzhou/datasets/imagenet/train/"
    ann_path = "/home/yangzhou/datasets/imagenet/meta/train.txt"
    datasets = ImgNet_C_train_Dst(img_path, ann_path)

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_uc = uc_loss().cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), args.lr, weight_decay=1e-3)
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 40, 70])

    epoch_p, itrs, acc, classifier, optimizer, lr_schedule = load_ckpt(classifier, optimizer, lr_schedule, args, "last")

    classifier.train()

    for epoch in range(epoch_p, 100):
        dataloader = tud.DataLoader(dataset=datasets,
                                    batch_size=args.bs,
                                    num_workers=32,
                                    pin_memory=True,
                                    shuffle=True
                                    )
        for i, (img, distort_img, ann, crp_idx, sev) in enumerate(dataloader):
            optimizer.zero_grad()
            f_clean, f_rec, theta, pred_clean, pred_rec = classifier(img.cuda(), distort_img.cuda())

            loss_uc = criterion_uc(f_clean, f_rec, theta)
            loss_cls1 = criterion(pred_clean, ann.cuda())
            loss_cls2 = criterion(pred_rec, ann.cuda())
            loss = 0.01 * loss_uc + loss_cls1 + loss_cls2
            loss.backward()
            optimizer.step()
            print(i, " loss: %.6f", loss)
            if (i + 1) % 625 == 0:
                if epoch % 2 == 0:
                    acc_val = val(classifier, args, "last")
                    if acc_val >= acc:
                        save_ckpt(epoch, itrs, classifier, optimizer, lr_schedule, acc_val, args, True)
                    else:
                        save_ckpt(epoch, itrs, classifier, optimizer, lr_schedule, acc_val, args, False)
                lr_schedule.step()
            itrs += 1


if __name__ == "__main__":
    setup_seed(0)
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    backbone = resnet50(pretrained=True)
    classifier = new_clsnet(backbone)
    classifier = torch.nn.DataParallel(classifier).cuda()

    if args.mode == "train":
        train(classifier, args)
    else:
        val(classifier, args, "best")

