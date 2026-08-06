#!/usr/bin/env python3
from __future__ import annotations
from collections import OrderedDict
import hashlib, json, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets.patch_sampler import AnatomyAwarePlanner

MASK_NAMES = [
    "Br1a","Br1b","Br2a","Br2b","CB1","CB2","CH1","CH2","CL1","CL2",
    "D1","D2","EN1","EN2","Hm1","Hm2","M1","M2","N","Oc1","Oc2",
    "Op1","Op2","P","VC"
]
TARGET_ANCHOR = {"Br2a":"Br1a", "Br2b":"Br1b"}

class FishDataset(Dataset):
    def __init__(self, root_dir, metadata_csv, indices=None, patch_size=None,
                 resize_to=(966,1288), transforms=None, training=False,
                 sampler_config=None):
        self.root_dir = Path(root_dir)
        self.meta = pd.read_csv(metadata_csv)
        if indices is not None:
            self.meta = self.meta.iloc[indices].reset_index(drop=True)
        self.patch_size = patch_size
        self.resize_to = resize_to
        self.transforms = transforms
        self.training = bool(training)
        self.cfg = sampler_config or {}
        self.rng = random.Random(int(self.cfg.get("seed",42)))
        self.cache_size = int(self.cfg.get("decoded_cache_size",2))
        self.decoded = OrderedDict()
        self.records = self.models = self.planner = None

        if self.training and patch_size is not None and self.cfg.get("enabled",False):
            self.records, self.models = self._load_or_build_cache()
            probs = {
                "target_positive": self.cfg.get("target_positive_prob",0.30),
                "target_hard_negative": self.cfg.get("target_hard_negative_prob",0.30),
                "general_foreground": self.cfg.get("general_foreground_prob",0.30),
                "random": self.cfg.get("random_prob",0.10),
            }
            self.planner = AnatomyAwarePlanner(
                self.records,
                self.cfg.get("target_structures",["Br2a","Br2b"]),
                probs,
                int(self.cfg.get("seed",42)),
            )


    def reseed_worker(self, seed):
        """Give every DataLoader worker an independent deterministic RNG stream."""
        seed = int(seed) % (2**32)
        self.rng.seed(seed)
        if self.planner is not None:
            self.planner.rng.seed(seed + 17)

    def __len__(self):
        if self.training and self.cfg.get("samples_per_epoch") is not None:
            return int(self.cfg["samples_per_epoch"])
        return len(self.meta)

    @staticmethod
    def load_mask(path):
        return (np.array(Image.open(path).convert("L"),dtype=np.uint8)>0).astype(np.uint8)

    @staticmethod
    def centroid(mask):
        ys,xs=np.where(mask>0)
        return None if len(xs)==0 else (float(np.median(xs)),float(np.median(ys)))

    @staticmethod
    def bbox(mask):
        ys,xs=np.where(mask>0)
        if len(xs)==0: return None
        return (float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max()))

    @staticmethod
    def p_frame(mask):
        ys,xs=np.where(mask>0)
        if len(xs)<3: return None
        pts=np.column_stack([xs,ys]).astype(float)
        centre=np.median(pts,axis=0)
        z=pts-centre
        vals,vecs=np.linalg.eigh(np.cov(z,rowvar=False))
        major=vecs[:,np.argsort(vals)[::-1][0]]
        minor=vecs[:,np.argsort(vals)[::-1][1]]
        if major[0]<0: major=-major
        if np.linalg.det(np.column_stack([major,minor]))<0: minor=-minor
        a=z@major; b=z@minor
        sa=float(np.percentile(a,95)-np.percentile(a,5))
        sb=float(np.percentile(b,95)-np.percentile(b,5))
        if sa<=1 or sb<=1: return None
        return {"centre":centre.tolist(),"major":major.tolist(),"minor":minor.tolist(),
                "scale_major":sa,"scale_minor":sb}

    @staticmethod
    def to_p(point,frame):
        p=np.asarray(point,float); c=np.asarray(frame["centre"],float)
        a=np.asarray(frame["major"],float); b=np.asarray(frame["minor"],float)
        d=p-c
        return (float(d@a/frame["scale_major"]),float(d@b/frame["scale_minor"]))

    @staticmethod
    def from_p(coord,frame):
        c=np.asarray(frame["centre"],float)
        a=np.asarray(frame["major"],float); b=np.asarray(frame["minor"],float)
        p=c+coord[0]*frame["scale_major"]*a+coord[1]*frame["scale_minor"]*b
        return (float(p[0]),float(p[1]))

    @staticmethod
    def robust_model(values):
        if not values: return None
        arr=np.asarray(values,float); med=np.median(arr,axis=0)
        return {"median":med.tolist(),
                "median_residual":float(np.median(np.linalg.norm(arr-med,axis=1))),
                "n":int(len(arr))}

    def _cache_path(self):
        d=Path(self.cfg.get("cache_dir","metadata/anatomy_sampler_cache"))
        d.mkdir(parents=True,exist_ok=True)
        payload={"ids":self.meta["image_id"].astype(str).tolist(),
                 "targets":self.cfg.get("target_structures",["Br2a","Br2b"]),
                 "version":3}
        h=hashlib.sha1(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:12]
        return d/f"anatomy_index_{h}.json"

    def _load_or_build_cache(self):
        path=self._cache_path()
        if path.exists() and not self.cfg.get("force_rebuild_cache",False):
            with path.open() as f: data=json.load(f)
            print(f"Loaded anatomy sampler cache: {path}")
            return data["records"],data["models"]
        print(f"Building training-only anatomy cache for {len(self.meta)} images...")
        records=self._scan()
        models=self._fit(records)
        self._fill_absent(records,models)
        with path.open("w") as f: json.dump({"records":records,"models":models},f,indent=2)
        print(f"Saved anatomy sampler cache: {path}")
        return records,models

    def _scan(self):
        targets=self.cfg.get("target_structures",["Br2a","Br2b"])
        records=[]
        for _,row in self.meta.iterrows():
            image_id=str(row["image_id"]); folder=self.root_dir/image_id
            centres={}; present=[]; masks={}
            for name in MASK_NAMES:
                m=self.load_mask(folder/f"{name}.png"); masks[name]=m
                c=self.centroid(m)
                if c is not None: present.append(name); centres[name]=list(c)
            full_path=folder/"full_mask.png"
            if full_path.exists(): full=self.load_mask(full_path)
            else:
                full=np.zeros_like(next(iter(masks.values())))
                for m in masks.values(): full=np.maximum(full,m)
            targets_info={}
            for t in targets:
                c=self.centroid(masks[t])
                targets_info[t]={"present":c is not None,
                                 "centroid":None if c is None else list(c),
                                 "expected_centroid":None,
                                 "localization_method":None}
            records.append({
                "image_id":image_id,
                "present_structures":present,
                "structure_centroids":centres,
                "full_mask_bbox":None if self.bbox(full) is None else list(self.bbox(full)),
                "p_frame":self.p_frame(masks["P"]),
                "targets":targets_info,
            })
        return records

    def _fit(self,records):
        models={}
        for target in self.cfg.get("target_structures",["Br2a","Br2b"]):
            anchor=TARGET_ANCHOR[target]
            offsets=[]; p_rel=[]; full_rel=[]
            for r in records:
                tc=r["targets"][target]["centroid"]
                if tc is None: continue
                frame=r["p_frame"]; ac=r["structure_centroids"].get(anchor)
                if frame is not None:
                    tp=self.to_p(tc,frame); p_rel.append(tp)
                    if ac is not None:
                        ap=self.to_p(ac,frame)
                        offsets.append((tp[0]-ap[0],tp[1]-ap[1]))
                box=r["full_mask_bbox"]
                if box is not None:
                    x1,y1,x2,y2=box
                    full_rel.append(((tc[0]-x1)/max(x2-x1,1),(tc[1]-y1)/max(y2-y1,1)))
            models[target]={
                "local_anchor":anchor,
                "br1_offset_in_p_frame":self.robust_model(offsets),
                "target_relative_to_p":self.robust_model(p_rel),
                "target_relative_to_full_mask":self.robust_model(full_rel),
            }
        return models

    def _estimate(self,r,target,m):
        candidates=[]
        frame=r["p_frame"]; anchor=r["structure_centroids"].get(m["local_anchor"])
        lm=m["br1_offset_in_p_frame"]
        if frame is not None and anchor is not None and lm is not None:
            ap=self.to_p(anchor,frame); off=lm["median"]
            candidates.append((self.from_p((ap[0]+off[0],ap[1]+off[1]),frame),
                               1/max(lm["median_residual"],1e-3),
                               f'{m["local_anchor"]}+P_frame'))
        pm=m["target_relative_to_p"]
        if frame is not None and pm is not None:
            candidates.append((self.from_p(tuple(pm["median"]),frame),
                               1/max(pm["median_residual"],1e-3),"P_relative"))
        if candidates:
            pts=np.asarray([x[0] for x in candidates]); w=np.asarray([x[1] for x in candidates])
            p=np.average(pts,axis=0,weights=w)
            return (float(p[0]),float(p[1])),"+".join(x[2] for x in candidates)
        fm=m["target_relative_to_full_mask"]; box=r["full_mask_bbox"]
        if fm is not None and box is not None:
            x1,y1,x2,y2=box; rel=fm["median"]
            return (x1+rel[0]*max(x2-x1,1),y1+rel[1]*max(y2-y1,1)),"full_mask_relative"
        return None,None

    def _fill_absent(self,records,models):
        for r in records:
            for t,m in models.items():
                if r["targets"][t]["present"]: continue
                c,method=self._estimate(r,t,m)
                r["targets"][t]["expected_centroid"]=None if c is None else list(c)
                r["targets"][t]["localization_method"]=method

    def _resize(self,image,masks):
        if self.resize_to is None: return image,masks
        h,w=self.resize_to
        image=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
        masks=np.stack([cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST) for m in masks])
        return image,(masks>0).astype(np.uint8)

    def _load_decoded(self,index):
        if index in self.decoded:
            image,masks,original_shape=self.decoded.pop(index)
            self.decoded[index]=(image,masks,original_shape)
            return image.copy(),masks.copy(),original_shape
        row=self.meta.iloc[index]; folder=self.root_dir/str(row["image_id"])
        image=np.array(Image.open(folder/"image.png").convert("RGB"),dtype=np.uint8)
        masks=np.stack([self.load_mask(folder/f"{n}.png") for n in MASK_NAMES])
        original_shape=masks.shape[1:]
        image,masks=self._resize(image,masks)
        if self.cache_size>0:
            self.decoded[index]=(image.copy(),masks.copy(),original_shape)
            while len(self.decoded)>self.cache_size: self.decoded.popitem(last=False)
        return image,masks,original_shape

    def make_sample_plan(self):
        if self.planner is None: raise RuntimeError("Planner is disabled.")
        return self.planner.make_plan().to_dict()

    def _crop_origin(self,cx,cy,h,w):
        p=int(self.patch_size); sigma=float(self.cfg.get("centre_jitter",64))/2
        cx+=self.rng.gauss(0,sigma); cy+=self.rng.gauss(0,sigma)
        x=min(max(0,int(round(cx-p/2))),w-p)
        y=min(max(0,int(round(cy-p/2))),h-p)
        return x,y

    def __getitem__(self,idx):
        plan=self.planner.make_plan() if self.planner is not None else None
        source=plan.source_index if plan is not None else idx%len(self.meta)
        image,masks,original_shape=self._load_decoded(source)
        row=self.meta.iloc[source]
        image=image.astype(np.float32)/255.0; masks=masks.astype(np.float32)
        meta={"requested_mode":"full_image","actual_mode":"full_image","sampling_mode":"full_image",
              "target_structure":"","localization_method":"none","fallback":0,
              "crop_x":-1,"crop_y":-1,"target_pixels_full":0,"target_pixels_patch":0,
              "target_fraction_retained":0.0}

        if self.patch_size is not None:
            h,w=image.shape[:2]
            if plan is None:
                union=masks.sum(axis=0)>0
                if self.rng.random()<0.9 and union.any():
                    ys,xs=np.where(union); k=self.rng.randrange(len(xs)); cx,cy=xs[k],ys[k]
                else: cx,cy=self.rng.randrange(w),self.rng.randrange(h)
            elif plan.centre_x is None:
                cx,cy=self.rng.randrange(w),self.rng.randrange(h)
            else:
                oh,ow=original_shape
                rh,rw=(h,w)
                cx=plan.centre_x*rw/ow; cy=plan.centre_y*rh/oh
            x,y=self._crop_origin(cx,cy,h,w); p=int(self.patch_size)
            full=patch=0
            if plan is not None and plan.target_structure in MASK_NAMES:
                ch=MASK_NAMES.index(plan.target_structure); full=int(masks[ch].sum())
            image=image[y:y+p,x:x+p,:]; masks=masks[:,y:y+p,x:x+p]
            if plan is not None and plan.target_structure in MASK_NAMES:
                patch=int(masks[ch].sum())
            meta={"requested_mode":"legacy_patch" if plan is None else plan.requested_mode,
                  "actual_mode":"legacy_patch" if plan is None else plan.actual_mode,
                  "sampling_mode":"legacy_patch" if plan is None else plan.actual_mode,
                  "target_structure":"" if plan is None else plan.target_structure,
                  "localization_method":"legacy_union" if plan is None else plan.localization_method,
                  "fallback":0 if plan is None else int(plan.fallback),
                  "crop_x":x,"crop_y":y,"target_pixels_full":full,
                  "target_pixels_patch":patch,
                  "target_fraction_retained":float(patch/max(full,1))}

        if self.transforms is not None:
            out=self.transforms(image=image,mask=masks.transpose(1,2,0))
            image=out["image"]; masks=out["mask"].transpose(2,0,1)
        image=image.transpose(2,0,1)
        result={"image":torch.from_numpy(image.astype(np.float32)),
                "mask":torch.from_numpy(masks.astype(np.float32)),
                "image_id":str(row["image_id"]),"fish_id":str(row["fish_id"]),
                "genotype":str(row["genotype"]),"quality":int(row["quality"])}
        result.update(meta)
        return result
