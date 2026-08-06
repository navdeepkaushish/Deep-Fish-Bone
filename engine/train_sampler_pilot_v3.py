#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from datetime import timedelta
from pathlib import Path
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from configs.structure_names import STRUCTURES
from datasets.metadata import Metadata
from engine.dataloaders import build_full_image_loader,build_patch_loaders
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image
from losses.combined import CombinedLoss
from models.unetpp import build_model
from utils.save_metrics import save_metrics_csv
from utils.seed import seed_everything

def load_json(path):
    with open(path) as f:return json.load(f)
def save_json(data,path):
    with open(path,"w") as f:json.dump(data,f,indent=2)
def norm(v):
    s=str(v).strip();return s[:-2] if s.endswith(".0") else s

def exact_split(df,manifest_path):
    m=pd.read_csv(manifest_path)
    lookup={norm(v):i for i,v in enumerate(df["image_id"])}
    result={}
    for split in ("train","val","test"):
        ids=m.loc[m["split"].astype(str).str.lower()==split,"image_id"].map(norm)
        result[split]=[lookup[x] for x in ids]
    fish={k:set(df.iloc[v]["fish_id"]) for k,v in result.items()}
    if not fish["train"].isdisjoint(fish["val"]) or not fish["train"].isdisjoint(fish["test"]) or not fish["val"].isdisjoint(fish["test"]):
        raise RuntimeError("Fish leakage detected.")
    return result["train"],result["val"],result["test"]

def checkpoint(path,epoch,model,opt,sched,score):
    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                "optimizer_state_dict":opt.state_dict(),
                "scheduler_state_dict":sched.state_dict(),
                "checkpoint_score":score},path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/focal_sampler_pilot_fold0_v3.json")
    args=ap.parse_args();cfg=load_json(args.config);seed_everything(cfg.get("seed",42))
    out=Path(cfg["output_root"])/"fold_0";ckpt=out/"checkpoints";ckpt.mkdir(parents=True,exist_ok=True)
    save_json(cfg,out/"config_used.json")
    df=Metadata(cfg["metadata_csv"]).dataframe().reset_index(drop=True)
    train_idx,val_idx,test_idx=exact_split(df,Path(cfg["baseline_split_manifest"]))
    print(f"Exact baseline split: train={len(train_idx)}, val={len(val_idx)}, untouched test={len(test_idx)}")
    train_loader,patch_val=build_patch_loaders(
        cfg["root_dir"],cfg["metadata_csv"],train_idx,val_idx,cfg["batch_size"],
        cfg["patch_size"],tuple(cfg["resize_to"]),cfg["num_workers"],cfg["sampler"])
    full_val=build_full_image_loader(
        cfg["root_dir"],cfg["metadata_csv"],val_idx,tuple(cfg["resize_to"]),cfg["full_valid_workers"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model().to(device);criterion=CombinedLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    sched=CosineAnnealingLR(opt,T_max=cfg["epochs"],eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    writer=SummaryWriter(str(out/"tensorboard"))
    best=float("-inf");best_epoch=-1;start=time.perf_counter()
    for epoch in range(cfg["epochs"]):
        print(f"\nPilot fold 0 | Epoch {epoch+1}/{cfg['epochs']}",flush=True)
        loss=train_one_epoch(model=model,loader=train_loader,criterion=criterion,
                             optimizer=opt,scaler=scaler,device=device)
        pvloss,pvdice,_,_=validate(model=model,loader=patch_val,criterion=criterion,
                                  device=device,epoch=epoch)
        fv=validate_full_image(
            model=model,loader=full_val,device=device,epoch=epoch,
            patch_size=cfg["patch_size"],stride=cfg["stride"],
            threshold=cfg["threshold"],min_component_pixels=cfg["min_component_pixels"],
            collect_presence=True,save_all_predictions=False,
            save_dir=out/"val_preview",return_details=True)
        sched.step()
        metrics={"epoch":epoch+1,"train_loss":loss,"patch_val_loss":pvloss,
                 "patch_mean_dice":pvdice,"full_val_mean_dice":fv["mean_dice"],
                 "lr":opt.param_groups[0]["lr"]}
        presence=fv.get("presence_summary")
        if presence is not None:
            for target in cfg["sampler"]["target_structures"]:
                r=presence.loc[presence["structure"]==target]
                if len(r)==1:
                    r=r.iloc[0];tp,fp,fn=int(r["tp"]),int(r["fp"]),int(r["fn"])
                    metrics.update({
                        f"full_val_{target}_tp":tp,f"full_val_{target}_fp":fp,
                        f"full_val_{target}_fn":fn,
                        f"full_val_{target}_precision":tp/max(tp+fp,1),
                        f"full_val_{target}_recall":tp/max(tp+fn,1),
                        f"full_val_{target}_f1":2*tp/max(2*tp+fp+fn,1)})
        save_metrics_csv(metrics,str(out/"metrics.csv"))
        writer.add_scalar("loss/train",loss,epoch);writer.add_scalar("dice/full_val",fv["mean_dice"],epoch)
        if fv["mean_dice"]>best:
            best=fv["mean_dice"];best_epoch=epoch+1
            checkpoint(ckpt/"best_model.pt",epoch,model,opt,sched,best)
            if presence is not None:presence.to_csv(out/"best_val_presence_summary.csv",index=False)
        print(f"  train loss    : {loss:.4f}")
        print(f"  full-val Dice : {fv['mean_dice']:.4f}")
        print(f"  best full-val : {best:.4f} (epoch {best_epoch})")
    writer.close()
    summary={"best_val_score":best,"best_epoch":best_epoch,
             "train_images":len(train_idx),"val_images":len(val_idx),
             "test_images_untouched":len(test_idx),"test_was_evaluated":False,
             "elapsed":str(timedelta(seconds=int(time.perf_counter()-start)))}
    save_json(summary,out/"pilot_summary.json")
    print("\nPilot complete. Independent test set was not evaluated.");print(summary)

if __name__=="__main__":main()
