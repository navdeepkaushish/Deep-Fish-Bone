# Deep Fish Bone Segmentation – Vast.ai Standard Operating Procedure (SOP)

## Phase 1. Create Vast.ai Instance

Recommended configuration

* GPU: RTX 3090 (24 GB) or RTX 4090
* Disk: ≥ 60 GB (preferably 80–100 GB)
* CUDA image with Python 3.12
* Jupyter + Terminal enabled

---

## Phase 2. Clone GitHub Repository

```bash
cd /workspace

git clone git@github.com:navdeepkaushish/Deep-Fish-Bone.git

cd Deep-Fish-Bone
```

If SSH cloning fails:

```bash
git clone https://github.com/navdeepkaushish/Deep-Fish-Bone.git
```

---

## Phase 3. Upload Dataset

Upload

```
ventral.zip
```

into

```
/workspace/Deep-Fish-Bone/
```

Extract

```bash
unzip ventral.zip
```

---

## Phase 4. Fix CL1 filename (Mac issue)

```bash
find ventral -name "Cl1.png" -execdir mv "Cl1.png" "CL1.png" \;
```

---

## Phase 5. Install Python Packages

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

## Phase 6. Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected

```
True
NVIDIA RTX 3090
```

---

## Phase 7. Verify Dataset

```bash
python - <<EOF
from pathlib import Path

required = [
"Br1a","Br1b","Br2a","Br2b","CB1","CB2",
"CH1","CH2","CL1","CL2","D1","D2",
"EN1","EN2","Hm1","Hm2","M1","M2",
"N","Oc1","Oc2","Op1","Op2","P","VC"
]

missing=[]

for folder in Path("ventral").iterdir():
    if folder.is_dir():
        for m in required:
            if not (folder/f"{m}.png").exists():
                missing.append((folder.name,m))

print("Missing =",len(missing))
print(missing[:20])
EOF
```

Expected

```
Missing = 0
```

---

## Phase 8. Run Smoke Test

```bash
PYTHONPATH=. python engine/train.py
```

Interrupt after one epoch

```
Ctrl+C
```

---

## Phase 9. Start Experiment

Example

```bash
PYTHONPATH=. python engine/train.py > train.log 2>&1
```

or

```bash
PYTHONPATH=. python engine/train_kfold.py \
--config configs/focal_cv.json \
> train_cv.log 2>&1
```

---

## Phase 10. Monitor Training

Terminal

```bash
tail -f train.log
```

GPU

```bash
watch -n 1 nvidia-smi
```

TensorBoard

Open another terminal

```bash
tensorboard \
--logdir outputs/tensorboard \
--host 0.0.0.0 \
--port 6007
```

---

## Phase 11. Save Experiment

Rename outputs

Example

```bash
mv outputs outputs_focal_lr3e4_bs2
```

Create experiment folder

```bash
mkdir -p experiments/Focal_lr3e4_bs2
```

Move files

```bash
mv outputs_focal_lr3e4_bs2 \
experiments/Focal_lr3e4_bs2/

mv train.log \
experiments/Focal_lr3e4_bs2/
```

---

## Phase 12. Compress Experiment

```bash
tar -czvf Focal_lr3e4_bs2.tar.gz \
experiments/Focal_lr3e4_bs2
```

---

## Phase 13. Download

Download

```
Focal_lr3e4_bs2.tar.gz
```

using the Jupyter file browser.

Verify locally

```bash
tar -tzf Focal_lr3e4_bs2.tar.gz | head
```

---

## Phase 14. After Download

Confirm archive is on local machine.

Then

* Stop instance.
* Destroy instance.

---

# Standard Training Settings

## BCE + Dice

```
batch_size = 2

learning_rate = 1e-4

num_workers = 6

full_validation_workers = 4

epochs = 100
```

---

## Focal + Dice

```
batch_size = 2

learning_rate = 3e-4

num_workers = 6

full_validation_workers = 4

epochs = 100
```

---

## Final Cross Validation

```
Loss = Focal + Dice

batch_size = 2

learning_rate = 3e-4

epochs = 100

5 folds
```
