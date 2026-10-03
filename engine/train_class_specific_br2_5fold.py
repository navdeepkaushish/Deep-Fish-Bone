
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from configs.structure_names import STRUCTURES
from datasets.metadata import Metadata
from engine.dataloaders import build_patch_loaders,build_full_image_loader
from engine.train_one_epoch import train_one_epoch
from engine.validate import validate
from engine.validate_full import validate_full_image
from losses.class_specific_br2 import ClassSpecificBr2Loss
from models.unetpp import build_model
from utils.save_metrics import save_metrics_csv
from utils.seed import seed_everything

def load_json(p):
    with open(p) as f:return json.load(f)
def save_json(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w") as f:json.dump(x,f,indent=2)
def norm(v):
    s=str(v).strip();return s[:-2] if s.endswith(".0") else s
def exact_split(df, path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    m = pd.read_csv(path)
    required = {"split", "image_id"}
    missing = required - set(m.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")

    metadata_ids = df["image_id"].map(norm)
    if metadata_ids.duplicated().any():
        raise RuntimeError("Duplicate image IDs in metadata.")
    lookup = {image_id: i for i, image_id in enumerate(metadata_ids)}

    out = {}
    for n in ("train", "val", "test"):
        ids = m.loc[m["split"].astype(str).str.lower() == n, "image_id"].map(norm).tolist()
        if not ids:
            raise RuntimeError(f"No {n} rows found in {path}")
        absent = [x for x in ids if x not in lookup]
        if absent:
            raise RuntimeError(f"{n} IDs absent from metadata: {absent[:20]}")
        out[n] = [lookup[x] for x in ids]

    train_fish = set(df.iloc[out["train"]]["fish_id"])
    val_fish = set(df.iloc[out["val"]]["fish_id"])
    test_fish = set(df.iloc[out["test"]]["fish_id"])
    if not train_fish.isdisjoint(val_fish):
        raise RuntimeError("Fish leakage train/val.")
    if not train_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage train/test.")
    if not val_fish.isdisjoint(test_fish):
        raise RuntimeError("Fish leakage val/test.")

    all_indices = out["train"] + out["val"] + out["test"]
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Image leakage between train/val/test subsets.")
    return out["train"], out["val"], out["test"]



EXPECTED_RECORDS = 190
EXPECTED_FISH = 34
EXCLUDED_IMAGE_IDS = {"525410710", "503027953"}
EXPECTED_FOLD_COUNTS = {
    0: (115, 37, 38, 21, 6, 7),
    1: (115, 38, 37, 21, 7, 6),
    2: (113, 39, 38, 20, 7, 7),
    3: (113, 38, 39, 20, 7, 7),
    4: (114, 38, 38, 20, 7, 7),
}

def preflight(cfg, df):
    metadata_path = ROOT / cfg["metadata_csv"]
    manifest_root = ROOT / cfg["manifest_root"]

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if len(df) != EXPECTED_RECORDS:
        raise RuntimeError(f"Expected {EXPECTED_RECORDS} metadata rows, found {len(df)}.")
    if "fish_id" not in df.columns or "image_id" not in df.columns:
        raise KeyError("Metadata must contain image_id and fish_id.")

    ids = df["image_id"].map(norm)
    if ids.duplicated().any():
        raise RuntimeError("Duplicate image_id values found in metadata.")
    if df["fish_id"].nunique() != EXPECTED_FISH:
        raise RuntimeError(
            f"Expected {EXPECTED_FISH} biological fish, found {df['fish_id'].nunique()}."
        )
    bad = sorted(set(ids) & EXCLUDED_IMAGE_IDS)
    if bad:
        raise RuntimeError(f"Excluded image IDs are still present: {bad}")

    all_test_ids = []
    for fold in range(5):
        path = manifest_root / f"fold_{fold}" / "split_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {path}")
        m = pd.read_csv(path)
        if len(m) != EXPECTED_RECORDS:
            raise RuntimeError(f"Fold {fold}: expected 190 manifest rows, found {len(m)}.")
        if {"split", "image_id"} - set(m.columns):
            raise KeyError(f"Fold {fold}: manifest must contain split and image_id.")

        mids = m["image_id"].map(norm)
        if mids.duplicated().any():
            raise RuntimeError(f"Fold {fold}: duplicate image_id values in manifest.")
        if set(mids) != set(ids):
            raise RuntimeError(f"Fold {fold}: manifest image IDs do not exactly match metadata.")

        # If the frozen manifest stores fish_id, verify it against corrected metadata.
        if "fish_id" in m.columns:
            fish_lookup = dict(zip(ids, df["fish_id"].astype(str)))
            mismatched = [
                image_id for image_id, fish_id in zip(mids, m["fish_id"].astype(str))
                if fish_lookup[image_id] != fish_id
            ]
            if mismatched:
                raise RuntimeError(
                    f"Fold {fold}: manifest fish_id disagrees with metadata for {mismatched[:20]}"
                )

        tr, va, te = exact_split(df, path)
        counts = (
            len(tr), len(va), len(te),
            df.iloc[tr]["fish_id"].nunique(),
            df.iloc[va]["fish_id"].nunique(),
            df.iloc[te]["fish_id"].nunique(),
        )
        if counts != EXPECTED_FOLD_COUNTS[fold]:
            raise RuntimeError(
                f"Fold {fold}: counts {counts}, expected {EXPECTED_FOLD_COUNTS[fold]}."
            )
        all_test_ids.extend(df.iloc[te]["image_id"].map(norm).tolist())
        print(
            f"Fold {fold}: train={counts[0]}/{counts[3]} fish, "
            f"val={counts[1]}/{counts[4]} fish, test={counts[2]}/{counts[5]} fish"
        )

    if len(all_test_ids) != EXPECTED_RECORDS or len(set(all_test_ids)) != EXPECTED_RECORDS:
        raise RuntimeError("Each of the 190 records must appear exactly once as test across folds.")

    print("Preflight PASS: frozen 190 records, 34 fish, five leakage-free rotations.")

def train_fold(fold,cfg,df,device):
    out=ROOT/cfg["output_root"]/f"fold_{fold}";ckdir=out/"checkpoints";ckdir.mkdir(parents=True,exist_ok=True)
    marker=out/"FOLD_COMPLETE.marker"
    if marker.exists() and not cfg.get("rerun_completed_folds",False):
        print(f"Fold {fold} complete; skipping."); return
    tr,va,te=exact_split(df,ROOT/cfg["manifest_root"]/f"fold_{fold}"/"split_manifest.csv")
    seed=int(cfg["seed"])+fold;seed_everything(seed)
    sc=json.loads(json.dumps(cfg["sampler"]));sc["seed"]=seed;sc["samples_per_epoch"]=len(tr)
    print(f"\nFold {fold+1}/5 train={len(tr)} val={len(va)} test={len(te)}")
    train_loader,patch_val_loader=build_patch_loaders(
        root_dir=str(ROOT/cfg["root_dir"]),metadata_csv=str(ROOT/cfg["metadata_csv"]),
        train_idx=tr,val_idx=va,batch_size=cfg["batch_size"],patch_size=cfg["patch_size"],
        resize_to=tuple(cfg["resize_to"]),num_workers=cfg["num_workers"],sampler_config=sc)
    full_val_loader=build_full_image_loader(
        root_dir=str(ROOT/cfg["root_dir"]),metadata_csv=str(ROOT/cfg["metadata_csv"]),
        indices=va,resize_to=tuple(cfg["resize_to"]),num_workers=cfg["full_valid_workers"])
    model=build_model().to(device);criterion=ClassSpecificBr2Loss(cfg["br2_alpha"],cfg["br2_beta"])
    print("Focal source:",criterion.focal_source)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["epochs"],eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1.;best_epoch=-1
    for epoch in range(cfg["epochs"]):
        tl=train_one_epoch(model=model,loader=train_loader,criterion=criterion,optimizer=opt,scaler=scaler,device=device)
        pvl,pmd,_,_=validate(model=model,loader=patch_val_loader,criterion=criterion,device=device,epoch=epoch)
        fv=validate_full_image(model=model,loader=full_val_loader,device=device,epoch=epoch,
            patch_size=cfg["patch_size"],stride=cfg["stride"],threshold=cfg["threshold"],
            min_component_pixels=cfg["min_component_pixels"],collect_presence=True,
            save_all_predictions=False,save_dir=out/"val_preview",return_details=True)
        sched.step()
        row={"fold":fold,"epoch":epoch+1,"train_loss":tl,"patch_val_loss":pvl,
             "patch_mean_dice":pmd,"full_val_mean_dice":fv["mean_dice"],"lr":opt.param_groups[0]["lr"]}
        for s,d,c in zip(STRUCTURES,fv["class_dice"],fv["class_counts"]):
            row[f"full_val_{s}_dice"]=d;row[f"full_val_{s}_count"]=int(c)
        save_metrics_csv(row,str(out/"metrics.csv"))
        if fv["mean_dice"]>best:
            best=float(fv["mean_dice"]);best_epoch=epoch+1
            torch.save({"fold":fold,"epoch":epoch,"model_state_dict":model.state_dict(),"checkpoint_score":best},
                       ckdir/"best_model.pt")
            if fv.get("presence_summary") is not None:
                fv["presence_summary"].to_csv(out/"best_val_presence_summary.csv",index=False)
        print(f"Fold {fold+1}/5 Epoch {epoch+1}/{cfg['epochs']} full_val={fv['mean_dice']:.4f} best={best:.4f}@{best_epoch}")
    ck=torch.load(ckdir/"best_model.pt",map_location=device)
    model.load_state_dict(ck["model_state_dict"],strict=True);model.eval()
    test_loader=build_full_image_loader(
        root_dir=str(ROOT/cfg["root_dir"]),metadata_csv=str(ROOT/cfg["metadata_csv"]),
        indices=te,resize_to=tuple(cfg["resize_to"]),num_workers=cfg["full_test_workers"])
    td=out/"independent_test";td.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        r=validate_full_image(model=model,loader=test_loader,device=device,epoch=int(ck["epoch"]),
            patch_size=cfg["patch_size"],stride=cfg["stride"],threshold=cfg["threshold"],
            min_component_pixels=cfg["min_component_pixels"],collect_presence=True,
            save_all_predictions=True,save_dir=td/"predictions",return_details=True)
    pd.DataFrame({"structure":STRUCTURES,"test_dice":r["class_dice"],"test_count":r["class_counts"]}).to_csv(td/"test_per_structure_dice.csv",index=False)
    r["presence_summary"].to_csv(td/"test_presence_summary.csv",index=False)
    r["presence_per_image"].to_csv(td/"test_presence_per_image.csv",index=False)
    summary={"fold":fold,"best_epoch":int(ck["epoch"])+1,"best_validation_dice":float(ck["checkpoint_score"]),
             "test_mean_dice":float(r["mean_dice"]),"n_train_images":len(tr),"n_val_images":len(va),"n_test_images":len(te)}
    save_json(summary,td/"test_summary.json")
    print(json.dumps(summary,indent=2))
    marker.write_text("complete\n")
    del model
    if device.type=="cuda":torch.cuda.empty_cache()

def aggregate(cfg):
    root=ROOT/cfg["output_root"];summ=[];dices=[];pres=[]
    for f in range(5):
        td=root/f"fold_{f}"/"independent_test"
        with open(td/"test_summary.json") as h:s=json.load(h)
        summ.append(s)
        d=pd.read_csv(td/"test_per_structure_dice.csv");d["fold"]=f;dices.append(d)
        p=pd.read_csv(td/"test_presence_summary.csv");p["fold"]=f;pres.append(p)
    sf=pd.DataFrame(summ).sort_values("fold");sf.to_csv(root/"fold_test_summary.csv",index=False)
    ad=pd.concat(dices,ignore_index=True);ap=pd.concat(pres,ignore_index=True)
    ad.to_csv(root/"all_folds_per_structure_dice.csv",index=False)
    ap.to_csv(root/"all_folds_presence_summary.csv",index=False)
    pr=[]
    for s,g in ap.groupby("structure",sort=False):
        tp,fp,fn,tn=map(int,[g.tp.sum(),g.fp.sum(),g.fn.sum(),g.tn.sum()])
        pr.append({"structure":s,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"total_errors":fp+fn,
                   "precision":tp/(tp+fp) if tp+fp else float("nan"),
                   "recall":tp/(tp+fn) if tp+fn else float("nan"),
                   "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float("nan")})
    pp=pd.DataFrame(pr);pp.to_csv(root/"pooled_presence_summary.csv",index=False)
    final={"mean_fold_test_dice":float(sf.test_mean_dice.mean()),
           "sample_sd_fold_test_dice":float(sf.test_mean_dice.std(ddof=1)),
           "fold_scores":sf.test_mean_dice.tolist()}
    save_json(final,root/"final_summary.json")
    print("\nFINAL SUMMARY\n",sf.to_string(index=False))
    print("\nPooled Br2\n",pp[pp.structure.isin(["Br2a","Br2b"])].to_string(index=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/class_specific_br2_5fold.json")
    ap.add_argument("--folds",nargs="*",type=int,default=None);ap.add_argument("--aggregate-only",action="store_true");ap.add_argument("--preflight-only",action="store_true")
    args=ap.parse_args();cfg=load_json(ROOT/args.config)
    df=Metadata(str(ROOT/cfg["metadata_csv"])).dataframe().reset_index(drop=True)
    preflight(cfg,df)
    if args.preflight_only:return
    if args.aggregate_only:return aggregate(cfg)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds=list(range(5)) if not args.folds else args.folds
    for f in folds:train_fold(f,cfg,df,device)
    if all((ROOT/cfg["output_root"]/f"fold_{f}"/"FOLD_COMPLETE.marker").exists() for f in range(5)):
        aggregate(cfg)
if __name__=="__main__":main()
