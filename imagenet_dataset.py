import torch.utils.data as tud
import cv2
from PIL import Image
import os
import torchvision.transforms as transforms
from imagecorruptions import corrupt, get_corruption_names
# from imagenet_c import corrupt
import random
import numpy as np
import torchvision.transforms.functional as TF

random.seed(0)


class ImgNet_C_val_Dst(tud.Dataset):
    def __init__(self, clean_img_path, img_path, ann_path, crp_name, severity):
        super(ImgNet_C_val_Dst, self).__init__()
        self.clean_img_path = clean_img_path
        self.img_path = img_path
        self.ann_path = ann_path
        self.crp_name = crp_name
        self.severity = severity
        self.image_ann_list = []
        with open(ann_path, 'r') as f:
            for i in f.readlines():
                self.image_ann_list.append(i)
        f.close()
        # print(self.image_ann_list[0])
        self.tsfrm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img_filename = self.image_ann_list[index].split(" ")[0]
        ann = int(self.image_ann_list[index].split(" ")[1])
        if self.crp_name == "clean":
            img = Image.open(os.path.join(self.clean_img_path, img_filename))
        else:
            img = Image.open(os.path.join(self.img_path, self.crp_name, self.severity, img_filename))
        img = img.convert("RGB")
        # img = cv2.imread(os.path.join(self.img_path, img_filename))
        # print(img)
        img = self.tsfrm(img)
        return img, ann

    def __len__(self):
        return len(self.image_ann_list)


class ImgNet_C_train_Dst(tud.Dataset):
    def __init__(self, img_path, ann_path):
        super(ImgNet_C_train_Dst, self).__init__()
        self.img_path = img_path
        self.ann_path = ann_path
        self.crp_name = get_corruption_names("common")
        self.sev_level = [1, 2, 3, 4, 5]
        self.image_ann_list = []
        with open(ann_path, 'r') as f:
            for i in f.readlines():
                self.image_ann_list.append(i)
        f.close()
        self.first_tsfrm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224)

        ])
        self.final_tsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(15)}

    def get_CrpNameAndSevLevel(self):
        id = random.randint(0, len(self.crp_sev_sample_dict.keys()) - 1)
        key = list(self.crp_sev_sample_dict.keys())[id]
        crp_func = self.crp_name[key]  # random select a corruption
        id = random.randint(0, len(self.crp_sev_sample_dict[key]) - 1)
        sev = self.crp_sev_sample_dict[key][id]  # random select a severity level
        self.crp_sev_sample_dict[key].pop(id)
        # keep all corruption and severity can be selected.
        if len(self.crp_sev_sample_dict[key]) == 0:
            self.crp_sev_sample_dict.pop(key)
        if len(self.crp_sev_sample_dict.keys()) == 0:
            self.crp_sev_sample_dict = {i: list(range(1, 6)) for i in range(15)}
        return crp_func, sev

    def __getitem__(self, index):
        img_filename = self.image_ann_list[index].split(" ")[0]
        ann = int(self.image_ann_list[index].split(" ")[1])
        img = Image.open(os.path.join(self.img_path, img_filename))
        img = img.convert("RGB")
        img = self.first_tsfrm(img)
        crp_func, sev = self.get_CrpNameAndSevLevel()

        img = np.array(img)
        distort_img = corrupt(img, severity=sev, corruption_name=crp_func)

        img = Image.fromarray(img)
        distort_img = Image.fromarray(distort_img)

        if random.random() > 0.5:
            img = TF.hflip(img)
            distort_img = TF.hflip(distort_img)

        img = self.final_tsfrm(img)
        distort_img = self.final_tsfrm(distort_img)
        return img, distort_img, ann, crp_func, sev

    def __len__(self):
        return len(self.image_ann_list)


if __name__ == "__main__":
    img_path = "/home/yangzhou/datasets/imagenet/train/"
    ann_path = "/home/yangzhou/datasets/imagenet/meta/train.txt"
    datasets = ImgNet_C_train_Dst(img_path, ann_path)
    dataloader = tud.DataLoader(dataset=datasets,
                                batch_size=1,
                                # num_workers=4,
                                # pin_memory=True,
                                shuffle=True
                                )
    print("load ok!")
    for i, (img, distort_img, ann, crp_func, sev) in enumerate(dataloader):
        print(img.size())
        print(ann)
        print(crp_func)
        print(sev)
        print(i)

        def show(img, name):
            import numpy as np
            img = np.asarray(img[0, :, :, :].permute(1,2,0))
            img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(name, img)


        show(img, "img")
        show(distort_img, crp_func[0]+" "+str(sev[0]))
        cv2.waitKey(0)
