#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2, numpy as np, torch
from configs.structure_names import STRUCTURES
from inference.sliding_window import sliding_window_predict
from models.unetpp import build_model

PROJECT=Path("/Users/navdeepkaushish/Documents/missing_bone_structures/deep-fish-bone")
CHECKPOINT=PROJECT/"outputs_cv_focal/fold_1/checkpoints/best_model.pt"
INPUT_ROOT=PROJECT/"ventral"
OUTPUT_ROOT=PROJECT/"ventral"

def load():
    ck=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    m=build_model()
    m.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    d=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    m.to(d).eval()
    return m,d

def main():
    p=argparse.ArgumentParser()
    p.add_argument("image_id")
    p.add_argument("structure")
    a=p.parse_args()
    if a.structure not in STRUCTURES: raise SystemExit("Unknown structure")
    img_path=INPUT_ROOT/a.image_id/"image.png"
    rgb=cv2.cvtColor(cv2.imread(str(img_path)),cv2.COLOR_BGR2RGB)
    H,W=rgb.shape[:2]
    rs=cv2.resize(rgb,(1288,966),interpolation=cv2.INTER_AREA)
    t=torch.from_numpy(rs).permute(2,0,1).float()/255.
    model,dev=load()
    probs=sliding_window_predict(model,t,dev,512,256)
    prob=probs[STRUCTURES.index(a.structure)].detach().cpu().numpy()
    prob=cv2.resize(prob,(W,H),interpolation=cv2.INTER_LINEAR)
    mask=(prob>=0.5).astype(np.uint8)
    n,l,s,_=cv2.connectedComponentsWithStats(mask,8)
    clean=np.zeros_like(mask)
    for i in range(1,n):
        if s[i,cv2.CC_STAT_AREA]>=25: clean[l==i]=1
    out=OUTPUT_ROOT/a.image_id
    out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/f"{a.structure}_mask.png"),clean*255)
    ov=rgb.copy(); sel=clean.astype(bool); red=np.zeros_like(rgb); red[...,0]=255
    ov[sel]=(0.55*rgb[sel]+0.45*red[sel]).astype(np.uint8)
    cv2.imwrite(str(out/f"{a.structure}_overlay.png"),cv2.cvtColor(ov,cv2.COLOR_RGB2BGR))
    print("Saved",out/f"{a.structure}_mask.png")
if __name__=="__main__": main()
