import os
import torch
from typing import List, Optional, Sequence, Union, Any, Callable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from copy import deepcopy
from torch.utils.data import Sampler
import numpy as np
from PIL import Image

BILINEAR = transforms.InterpolationMode.BILINEAR

class RandomRotation90(transforms.RandomRotation):
    def __init__(self, interpolation=BILINEAR):
        super().__init__(90, interpolation=interpolation)

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        angle = float(np.random.choice([0,90.0,-90.0],1).item())
        return angle
    
class meta_batchsampler(Sampler):
    
    def __init__(self,data_source,way,shots):

        self.way = way
        self.shots = shots

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id


    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])       

        while len(temp_class2id) >= self.way:

            id_list = []

            list_class_id = list(temp_class2id.keys())

            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            batch_class_id = np.random.choice(list_class_id,size=self.way,replace=False,p=pcount/sum(pcount))

            """ for shot in self.shots:
                    for class_id in batch_class_id:
                        for _ in range(shot):
                            id_list.append(temp_class2id[class_id].pop()) """
            
            for class_id in batch_class_id:
                for _ in range(sum(self.shots)):
                    id_list.append(temp_class2id[class_id].pop())

            for class_id in batch_class_id:
                if len(temp_class2id[class_id])<sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list

# sampler used for meta-testing
class random_sampler(Sampler):

    def __init__(self,data_source,way,shot,query_shot,trial=1000):

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = query_shot

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot
        
        class2id = deepcopy(self.class2id)        
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []
 
            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])

            """ for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot+query_shot)]) """
            
            for cat in picked_class:
                id_list.extend(class2id[cat][:shot+query_shot])
            
            yield id_list
        
    def __len__(self):
        return self.trial

class transformer_manage():

    def __init__(self) -> None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.cen_crop = [transforms.Resize(92),
                         transforms.CenterCrop(84),]
        self.flip_color = [transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4)]
        self.normlize = [transforms.ToTensor(),
                        transforms.Normalize(mean, std)]
        self.rotation = [RandomRotation90()]
        self.eras =  lambda p,mins,maxs: \
                        transforms.RandomErasing(p,scale=(mins, maxs),ratio=(0.3,3.3),value='random')
        self.crop_aug = lambda scrop,mins,maxs: \
                        transforms.RandomResizedCrop(scrop, scale=(mins, maxs))
        
    def get_train_trans(self,crop_list,is_rotation=False,is_flip_color=True,is_eras=False):
        scrop,mins,maxs = crop_list
        aug_list = [self.crop_aug(scrop,mins,maxs)]
        if is_flip_color:
            aug_list = self.flip_color + aug_list
        if is_rotation:
            aug_list = aug_list + self.rotation
        aug_list = aug_list + self.normlize

        if is_eras:
            aug_list = aug_list + [self.eras(0.5, 0.1, 0.4)]

        return transforms.Compose(aug_list)
    
    def get_test_trans(self,is_center=False,is_rcrop1=False,
                        is_rcrop2=False,is_hflip=True,is_eras=False,is_rotation=False):

        aug_list = self.normlize
        if is_center:
            aug_list = self.cen_crop + aug_list
        if is_rcrop1:
            aug_list = [transforms.RandomResizedCrop(84,(0.5, 0.83),(1,1))] + aug_list
        if is_rcrop2:
            aug_list = [transforms.RandomResizedCrop(84,(0.25, 0.7),(1,1))] + aug_list
        if is_hflip:
            aug_list = [self.flip_color[0]] + aug_list
        if is_eras:
            aug_list = aug_list + [self.eras(1.0,0.06,0.2)]
        
        return transforms.Compose(aug_list)
    
class trans_saveImg():

    def __init__(self) -> None:

        self.path = 'img'
    
    def __call__(self, img) -> Any:

        # 设置图片要被保存的路径和名称
        i = int(np.random.randint(3000,size=1).item())
        if isinstance(img,Image.Image):
            save_path = os.path.join('img', f'image_{i}.png')
            # 保存图片
            img.save(save_path)

class CustomCropTransform():
    def __init__(self, scale):
        self.scale = scale
        self.rescale = transforms.Resize((84, 84))  # 将尺寸更改为原始尺寸，这里以32x32为例

        trans_manage = transformer_manage()
        self.norm = transforms.Compose([trans_manage.flip_color[0]]+trans_manage.normlize)
        self.center = transforms.Compose(trans_manage.cen_crop)


    def __call__(self, img):
        w, h = img.size
        img_crops = []
        short_e = min(w, h)
        
        c_short_e = short_e * self.scale
        regions = {
            'topleft': img.crop((0, 0, c_short_e, c_short_e)),
            'topright': img.crop((w-c_short_e, 0, w, c_short_e)),
            'bottomleft': img.crop((0, h-c_short_e, c_short_e, h)),
            'bottomright': img.crop((w-c_short_e, h-c_short_e, w, h)),
        }
        c_img = self.center(img)
        img_crops.append(self.norm(c_img))
        for k in regions:
            crop_resize_img = self.rescale(regions[k])
            img_crops.append(self.norm(crop_resize_img))

        return img_crops


class TestAug:
    def __init__(self, 
                 num_aug=3):
        trans_manage = transformer_manage()
        trans = []

        for i in range(num_aug):
            if i == 0:
                trans.append(trans_manage.get_test_trans(is_center=True))
            elif i == 1:
                trans.append(trans_manage.get_test_trans(is_rcrop1=True))
            elif i == 2:
                trans.append(trans_manage.get_test_trans(is_rcrop2=True))
            else:
                assert False," 错咯错咯 i 超出范围"

        self.trans = trans

    def __call__(self, img):

        img_crops = []
        for trans in self.trans:
            crop = trans(img)
            img_crops.append(crop)

        return img_crops

class MyDataAug:
    def __init__(self, 
                 size_crops=(84,48,48), 
                 min_scale_crops=(0.2, 0.08, 0.08), 
                 max_scale_crops=(1, 0.2, 0.2), 
                 num_crops=[1, 1]):
        trans_manage = transformer_manage()
        trans = []

        for i in range(len(num_crops)):
            crop_list = [size_crops[i],min_scale_crops[i],max_scale_crops[i]]
            if i == 0:
                trans.append(trans_manage.get_train_trans(crop_list))
            elif i == 1:
                 trans.append(trans_manage.get_train_trans(crop_list))
            elif i == 2:
                 trans.append(trans_manage.get_train_trans(crop_list))
            else:
                assert False," 错咯错咯 i 超出范围"

        self.trans = trans

    def __call__(self, img):

        img_crops = []
        for trans in self.trans:
            crop = trans(img)
            img_crops.append(crop)

        return img_crops

def my_collate(batch):
    #batch： [(img_crops1,target1),(img_crops2,target2)..]
    num_crops = len(batch[0][0])
    #data_crops_list: num_crop x [img]*batchsize
    data_crops_list, target_list = [[] for _ in range(num_crops)],[]
    for item in batch:
        for i in range(num_crops):
            data_crops_list[i].append(item[0][i])
        target_list.append(torch.tensor(item[1]))
    data_crops = [torch.stack(d) for d in data_crops_list] 
    target = torch.stack(target_list).repeat(num_crops)
    return data_crops, target