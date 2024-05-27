import os
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transforms_manager import *


class FewDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_way: int = 10,
        shot: int = 5,
        query_shot: int= 10,
        trail: int =400,
        image_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        num_crops = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_way = train_way
        self.shot = shot
        self.query_shot = query_shot
        self.trail = trail
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_crops = num_crops


    def setup(self, stage: Optional[str] = None) -> None:

        
        transform_val = transforms.Compose([transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        normalaug = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                        transforms.RandomResizedCrop(self.image_size),
                                        transform_val])

        train_transform = transforms.Compose([#MyDataAug(num_crops=self.num_crops),
                                              normalaug,
                                              ])
        
        self.train_dataset = ImageFolder(os.path.join(self.data_dir,'images'), transform = train_transform)#cut_train
        #self.val_dataset = ImageFolder(os.path.join(self.data_dir,'test_pre'), transform = transform_val)
        self.val_dataset = ImageFolder(os.path.join("/home/czd/Dataset/CUB_100_50_50/",'test_pre'), transform=transform_val)

#      ===============================================================
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_sampler = meta_batchsampler(self.train_dataset, self.train_way, [self.shot,self.query_shot]),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            #collate_fn = my_collate,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [DataLoader(self.val_dataset,
                            batch_sampler = random_sampler(data_source=self.val_dataset,way=5,shot=5,
                                                   query_shot=15,trial=self.trail),
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory),

                DataLoader(self.val_dataset,
                            batch_sampler = random_sampler(data_source=self.val_dataset,way=5,shot=1,
                                                   query_shot=15,trial=self.trail),
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory),]
    

def test_loader(
        data_path: str,
        test_way: int = 5,
        num_workers: int = 20,
        **kwargs):

    CUB_transform_test = transforms.Compose([transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    test_dataset = ImageFolder(os.path.join(data_path,'test_pre'), transform=CUB_transform_test)
    loader = [DataLoader(test_dataset,
                        batch_sampler = random_sampler(data_source=test_dataset,way=test_way,shot=5,
                                               query_shot=15,trial=3000),
                        num_workers = num_workers),
            DataLoader(test_dataset,
                        batch_sampler = random_sampler(data_source=test_dataset,way=test_way,shot=1,
                                               query_shot=15,trial=4000),
                        num_workers = num_workers)]
    return loader

def test_loader2(
        data_path: str,
        test_way: int = 5,
        num_workers: int = 20,
        test_scale: int = 0.75,
        **kwargs):
    #CustomCropTransform TestAug()
    print(f"MRR hyper-param:{test_scale}")
    test_transform = transforms.Compose([CustomCropTransform(scale=test_scale),]) #cub 77
    test_dataset = ImageFolder(os.path.join(data_path,'test'), transform=test_transform)
    loader = [DataLoader(test_dataset,
                        batch_sampler = random_sampler(data_source=test_dataset,way=test_way,shot=5,
                                               query_shot=15,trial=3000),
                        num_workers = num_workers,
                        collate_fn = my_collate,),
            DataLoader(test_dataset,
                        batch_sampler = random_sampler(data_source=test_dataset,way=test_way,shot=1,
                                               query_shot=15,trial=4000),
                        num_workers = num_workers,
                        collate_fn = my_collate),]
    return loader


class NDataset(LightningDataModule):
    """
    PyTorch Lightning data module 
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        trail :int,
        image_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_crops = None,
        **kwargs,
    ):
        super().__init__()

        self.trail = trail
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_crops = num_crops

    def setup(self, stage: Optional[str] = None) -> None:

        transform_val = transforms.Compose([transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        
        normalaug = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                        transforms.RandomResizedCrop(self.image_size),
                                        transform_val])
        
        if sum(self.num_crops) > 1:
            trans_aug = MyDataAug(num_crops=self.num_crops)
            self.collate_fn = my_collate
        else:
            trans_aug = normalaug
            self.collate_fn = None
        
        train_transform = transforms.Compose([trans_aug])

        self.train_dataset = ImageFolder(os.path.join(self.data_dir,'images'), transform=train_transform)
        self.val_dataset = ImageFolder(os.path.join(self.data_dir,'test_pre'), transform=transform_val)
       
#      ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn = self.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [DataLoader(self.val_dataset,
                            batch_sampler = random_sampler(data_source=self.val_dataset,way=5,shot=5,
                                                   query_shot=15,trial=self.trail),
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory),

                DataLoader(self.val_dataset,
                            batch_sampler = random_sampler(data_source=self.val_dataset,way=5,shot=1,
                                                   query_shot=15,trial=self.trail),
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory),]
    
    