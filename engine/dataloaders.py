#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader,get_worker_info
from datasets.fish_dataset import FishDataset
from datasets.transforms import get_train_transforms,get_valid_transforms

def _seed_worker(worker_id):
    info=get_worker_info()
    if info is not None and hasattr(info.dataset,"reseed_worker"):
        info.dataset.reseed_worker(torch.initial_seed())


def _dataset(root_dir,metadata_csv,indices,patch_size,resize_to,training,sampler_config=None):
    return FishDataset(
        root_dir=root_dir,metadata_csv=metadata_csv,indices=indices,
        patch_size=patch_size,resize_to=resize_to,
        transforms=get_train_transforms() if training else get_valid_transforms(),
        training=training,sampler_config=sampler_config if training else None)

def build_patch_loaders(root_dir,metadata_csv,train_idx,val_idx,batch_size=4,
                        patch_size=512,resize_to=(966,1288),num_workers=0,
                        sampler_config=None):
    train=_dataset(root_dir,metadata_csv,train_idx,patch_size,resize_to,True,sampler_config)
    val=_dataset(root_dir,metadata_csv,val_idx,patch_size,resize_to,False,None)
    common={"batch_size":batch_size,"num_workers":num_workers,
            "pin_memory":torch.cuda.is_available(),"worker_init_fn":_seed_worker}
    if num_workers>0:
        common.update(persistent_workers=True,prefetch_factor=2)
    return (DataLoader(train,shuffle=False,**common),
            DataLoader(val,shuffle=False,**common))

def build_full_image_loader(root_dir,metadata_csv,indices,resize_to=(966,1288),num_workers=0):
    dataset=_dataset(root_dir,metadata_csv,indices,None,resize_to,False,None)
    kwargs={"batch_size":1,"shuffle":False,"num_workers":num_workers,
            "pin_memory":torch.cuda.is_available(),"worker_init_fn":_seed_worker}
    if num_workers>0:
        kwargs.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(dataset,**kwargs)

def build_dataloaders(*args,**kwargs):
    return build_patch_loaders(*args,**kwargs)

def build_full_valid_loader(root_dir,metadata_csv,valid_idx,resize_to=(966,1288),num_workers=0):
    return build_full_image_loader(root_dir,metadata_csv,valid_idx,resize_to,num_workers)
