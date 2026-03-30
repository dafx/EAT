# Pre-training EAT from Scratch on FSD50K

## Overview

EAT is pre-trained using a self-supervised masked audio modeling objective (no labels needed).
Pre-training only requires `.tsv` manifest files and audio files — **no `.lbl` label files**.

---

## Tasks

### 1. Download & Prepare FSD50K Audio

- Download FSD50K from [Zenodo](https://zenodo.org/record/4060432)
- Convert all audio clips to `.wav` at **32kHz** (pre-training uses 32kHz, not 16kHz)
- Filter out clips shorter than ~0.31 seconds (10,000 samples at 32kHz) — they will be skipped anyway

```bash
# Example conversion with ffmpeg
ffmpeg -i input.flac -ar 32000 output.wav
```

---

### 2. Create TSV Manifest Files

Pre-training only reads `.tsv` files. Format:
- **Line 1**: Root directory (absolute path to audio files)
- **Lines 2+**: `<relative_file_path> <sample_count>` (whitespace-separated)

**Sample count** = number of audio samples (e.g., 320000 = 10 sec at 32kHz)

```
/path/to/fsd50k/audio
FSD50K.dev_audio/clip_001.wav 320000
FSD50K.dev_audio/clip_002.wav 105600
FSD50K.eval_audio/clip_003.wav 245760
```

Create `train.tsv` (dev set) and `valid.tsv` (eval set) in a single directory, e.g. `/data/fsd50k/`.

**Script to generate TSV:**
```python
import os, soundfile as sf

root = "/path/to/fsd50k/audio"
splits = {"train": "FSD50K.dev_audio", "valid": "FSD50K.eval_audio"}

for split, subdir in splits.items():
    audio_dir = os.path.join(root, subdir)
    with open(f"{split}.tsv", "w") as f:
        f.write(root + "\n")
        for fname in sorted(os.listdir(audio_dir)):
            if fname.endswith(".wav"):
                fpath = os.path.join(audio_dir, fname)
                info = sf.info(fpath)
                n_samples = int(info.frames * 32000 / info.samplerate)
                rel_path = os.path.join(subdir, fname)
                f.write(f"{rel_path}\t{n_samples}\n")
```

Place `train.tsv` and `valid.tsv` in a single data directory (e.g. `/data/fsd50k/`).

---

### 3. Create Pre-training Config

Copy the existing config and update the data path:

```bash
cp EAT/config/pretraining_AS2M.yaml EAT/config/pretraining_FSD50K.yaml
```

Edit `EAT/config/pretraining_FSD50K.yaml` — change the `task.data` field:

```yaml
task:
  _name: mae_image_pretraining
  data: /data/fsd50k          # <-- update this path
  rebuild_batches: true
  key: source
  precompute_mask_config: {}
  downsr_16hz: false          # pre-training uses 32kHz natively
  audio_mae: true
  h5_format: false            # set true if using HDF5 format
  target_length: 1024
  flexible_mask: false
```

Other parameters to consider adjusting (from the default AS-2M config):

| Parameter | AS-2M default | FSD50K suggestion |
|---|---|---|
| `optimization.max_update` | 400000 | 100000–200000 (dataset is ~50K clips) |
| `dataset.batch_size` | 12 | 12–24 |
| `optimization.lr` | `[0.0005]` | `[0.0005]` (keep or reduce slightly) |
| `optimizer.groups.default.lr_scheduler.warmup_updates` | 53333 | 10000–20000 |

---

### 4. Create Pre-training Script

```bash
cp EAT/scripts/pretraining_AS2M.sh EAT/scripts/pretraining_FSD50K.sh
```

Edit `EAT/scripts/pretraining_FSD50K.sh`:

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name pretraining_FSD50K \
    common.user_dir=EAT \
    checkpoint.save_dir=/path/to/checkpoints/fsd50k_pretrain \
    checkpoint.restore_file=/path/to/checkpoints/fsd50k_pretrain/checkpoint_last.pt \
    distributed_training.distributed_world_size=4 \
    dataset.batch_size=12 \
    task.data=/data/fsd50k \
    task.h5_format=False
```

---

### 5. Handle Normalization (Optional but Recommended)

The default normalization constants in `data/raw_audio_dataset.py` are computed from AudioSet:

```python
self.norm_mean = -4.268
self.norm_std = 4.569
```

For best results, compute FSD50K-specific stats and update these values in `data/raw_audio_dataset.py`:

```python
import numpy as np, torchaudio

means, stds = [], []
for fpath in fsd50k_wav_files:
    waveform, sr = torchaudio.load(fpath)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr,
        use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0.0, frame_shift=10
    )
    means.append(fbank.mean().item())
    stds.append(fbank.std().item())

print(np.mean(means), np.mean(stds))
```

Then edit `data/raw_audio_dataset.py` (around line 415):
```python
self.norm_mean = <fsd50k_mean>
self.norm_std  = <fsd50k_std>
```

---

### 6. Run Pre-training

```bash
bash EAT/scripts/pretraining_FSD50K.sh
```

Monitor with:
```bash
tail -f /path/to/checkpoints/fsd50k_pretrain/train.log
```

---

## File Checklist

| File | Action |
|---|---|
| `EAT/config/pretraining_FSD50K.yaml` | **Create** — copy from `pretraining_AS2M.yaml`, update `task.data` and training schedule |
| `EAT/scripts/pretraining_FSD50K.sh` | **Create** — copy from `pretraining_AS2M.sh`, update paths and config name |
| `/data/fsd50k/train.tsv` | **Create** — manifest for FSD50K dev set |
| `/data/fsd50k/valid.tsv` | **Create** — manifest for FSD50K eval set |
| `data/raw_audio_dataset.py` | **Modify** — update `norm_mean` / `norm_std` with FSD50K stats (optional) |

No other code changes are required — the model architecture, data pipeline, and masking strategy work as-is.

---

## Notes

- **No label files needed** for pre-training (self-supervised objective)
- **Audio must be 32kHz** (or set `downsr_16hz=true` to let the code downsample from higher rates)
- Clips shorter than ~0.31 sec (10,000 samples) are automatically skipped
- Clips longer than ~10 sec (325,000 samples) are randomly cropped
- After pre-training, fine-tune on FSD50K or other downstream tasks using `scripts/finetuning_*.sh`
