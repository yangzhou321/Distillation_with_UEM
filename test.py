import csv
import torch.utils.data as tud
import torch, os, argparse
from tqdm import tqdm
import numpy as np
from imagenet_dataset import ImgNet_C_val_Dst
from torchvision.models import resnet50
import torch.nn as nn


def parse():
    parser = argparse.ArgumentParser(description="train gurie")
    parser.add_argument("--bs", type=int, default=512, help="batch_size")
    parser.add_argument("--gpuids", type=str, default="0,1,2,3", help="GPU id to train")
    parser.add_argument("--method", type=str, default="R50v_mem", help="name of method")
    parser.add_argument("--mode", type=str, default="train", help="train_or_val")
    parser.add_argument("--root_path", type=str, default="/home/yangzhou/datasets/imagenet/", help="data root path")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/",
                        help="ckpt_path to load/val")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--nb", type=int, default=16, help="number works")
    parser.add_argument("--test_dst", type=str, default="c", help="test datasets")
    arg = parser.parse_args()
    return arg


def val(classifier, args):
    result_dict = {}
    ann_path = "/home/yangzhou/datasets/imagenet/meta/val.txt"
    clean_img_path = "/home/yangzhou/datasets/imagenet/ILSVRC2012_img_val/"
    img_path = "/home/yangzhou/datasets/imagenet_c/"
    # ann_path = "F:/ILSVRC2012/meta/val.txt"
    # clean_img_path = "F:/ILSVRC2012/ILSVRC2012_img_val/"
    # img_path = "H:/ImageNet_C"
    # classifier.load_state_dict(torch.load(ckpt_path)["f_net"])

    classifier.eval()
    # severity_level = ["1"]
    with open("result_new.csv", 'a') as fcsv:
        crp_name = ["gaussian_noise", "shot_noise", "impulse_noise",
                    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                    "snow", "frost", "fog",
                    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise",
                    "spatter", "gaussian_blur", "saturate"]
        severity_level = ["1", "2", "3", "4", "5"]
        f_csv = csv.writer(fcsv)
        # f_csv.writerow(crp_name)
        result = [args.method]
        for crp in crp_name:
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
                        pred = classifier(img.cuda())
                        hit = np.count_nonzero(np.argmax(pred.detach().cpu().numpy(), axis=1) == ann.numpy())
                        acc += hit
                print(crp, ", ", severity, ", Acc = ", acc / len(datasets))  # 72.648%  76.55%
                if crp == "clean":
                    avg_acc = acc / len(datasets)
                    break
                avg_acc += acc / len(datasets) / 5
            result_dict[str(crp)] = avg_acc
            result.append(1 - avg_acc)
        f_csv.writerow(result)
    fcsv.close()
    print(result_dict)


if __name__ == "__main__":
    args = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids

    model = resnet50(pretrained=True)
    model = nn.DataParallel(model).cuda()
    load_path = os.path.join(args.ckpt_path, args.method)
    load_path = os.path.join(load_path, "best.pth")

    state_dict = torch.load(load_path)
    model.module.load_state_dict(state_dict["model"])
    val(model, args)