import os
import torchvision
from torch.utils.data import DataLoader
from collections import OrderedDict

from core.dataset import Video_Dataset
from core.dataset.transform import *

def get_transforms(cfg, modality, mode="test"):
    transforms = OrderedDict()
    for m in modality:
        if m == "RGB":
            if mode == "train":
                transforms[m] = torchvision.transforms.Compose(
                    [
                        MultiScaleCrop(cfg.data.train_crop_size, [1, 0.875, 0.75, 0.66]),
                        RandomHorizontalFlip(prob=0.5),
                        Stack(m),
                        ToTensor(),
                        Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                    ]
                )
            else:
                transforms[m] = torchvision.transforms.Compose(
                    [
                        Rescale(cfg.data.test_scale_size),
                        CenterCrop(cfg.data.test_crop_size),
                        Stack(m),
                        ToTensor(),
                        Normalize(cfg.data.rgb.mean, cfg.data.rgb.std),
                    ]
                )
        elif m == "Flow":
            if mode == "train":
                transforms[m] = torchvision.transforms.Compose(
                    [
                        MultiScaleCrop(cfg.data.train_crop_size, [1, 0.875, 0.75]),
                        RandomHorizontalFlip(prob=0.5),
                        Stack(m),
                        ToTensor(),
                        Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                    ]
                )
            else:
                transforms[m] = torchvision.transforms.Compose(
                    [
                        Rescale(cfg.data.test_scale_size),
                        CenterCrop(cfg.data.test_crop_size),
                        Stack(m),
                        ToTensor(),
                        Normalize(cfg.data.flow.mean, cfg.data.flow.std),
                    ]
                )
        elif m == "Audio":
                transforms[m] = torchvision.transforms.Compose(
                    [Stack(m), ToTensor(is_audio=True)]
                )
                
    return transforms
    

def create_dataloader(cfg, logger, modality, mode="test"):
    logger.info(f"Creating {mode} Dataloader...")
    if mode == "train":
        vid_file = cfg.train.vid_list
        annotation_file = cfg.train.annotation_file
        batch_size = cfg.train.batch_size
        shuffle = True
    elif mode == "val":
        vid_file = cfg.val.vid_list
        annotation_file = cfg.train.annotation_file
        batch_size = cfg.val.batch_size
        shuffle = False
    elif mode == "test":
        vid_file = cfg.test.vid_list
        annotation_file = cfg.test.annotation_file
        batch_size = cfg.test.batch_size
        shuffle = False
        
        
    if vid_file:
        file_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        with open(os.path.join(file_dir, vid_file)) as f:
            vid_list = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        
    transforms=get_transforms(cfg, modality, mode)
    
    dataset = Video_Dataset(
        cfg,
        vid_list,
        annotation_file,
        modality,
        transform=transforms,
        mode=mode,
    )
    
    if mode == "train":
        batch_size = cfg.train.batch_size
    elif mode == "val":
        batch_size = cfg.val.batch_size
    elif mode == "test":
        batch_size = cfg.test.batch_size
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
    )
    logger.info("Done.")
    logger.info("----------------------------------------------------------")
    
    return dataloader