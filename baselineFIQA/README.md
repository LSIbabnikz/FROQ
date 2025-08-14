# Baseline FIQA (Auxiliary Perturbation Approach)

This repository implements the *auxiliary FIQA* procedure used to generate pseudo-quality labels for face images. The method follows the paper’s “Auxiliary FIQA Approach”: quality is estimated by **observing how a face recognition (FR) model’s embeddings change under small, controlled perturbations**, then averaging the similarity drops across three perturbations—**horizontal flip**, **Gaussian noise**, and **grid occlusion**. Formally, the final pseudo-quality score is the mean of the flip/noise/occlusion components computed via cosine similarity between embeddings of the original and perturbed images.

---

## Repository structure

> Everything lives in `baselineFIQA/` (paths below assume this subfolder).

- **`main.py`** – Orchestrates the full pipeline.  
  Loads config, instantiates the FR model and transform, then computes **flip**, **occlusion**, and **noise** scores with cosine similarity and averages them into a single dictionary `{image_path: quality_score}`. Also writes temporary per-component PKLs and the final combined labels to `cfg["outpath"]`. Uses `utils.parse_config_file` and `utils.load_fr_model`.

- **`dataset.py`** – Minimal dataset wrapper.  
  Recursively walks a root folder and collects image paths, applying the provided transform at access time. Base class used by all components.

- **`flip_component.py`** – Horizontal-flip component.  
  Defines a dataset that returns both original and left–right flipped images. `generate_flip_scores(...)` runs the FR model in eval/AMP on GPU, computes cosine similarity between original and flipped embeddings, and returns `{path: score}`.

- **`occlusion_component.py`** – Grid-occlusion component.  
  Builds a bank of binary masks (size controlled by `mpx` and `image_size`) and applies them across the image. `generate_occlusion_scores(...)` averages cosine similarities between the original embedding and all occluded variants to produce a score per image.

- **`noise_component.py`** – Gaussian-noise component.  
  For each image, samples several random noise masks and iterates over a user-supplied list of noise strengths `alphas`. For each (mask, α), it computes embeddings for the noisy mix and averages the cosine similarities to get the final noise score.

- **`baseline_FIQA_config.yaml`** – Example configuration.  
  Consumed by `main.py`. Expected keys (see code):  
  - `face_recognition_config`: path to a config that `utils.load_fr_model(...)` understands  
  - `dataset`:  
    - `path`: root folder of images (recursively scanned)  
    - `batch_size`: dataloader batch size  
    - `mpx`: pixel size of occlusion squares  
    - `image_size`: input spatial size to which masks are shaped (e.g., 112)  
    - `alphas`: list of floats for noise mixing strengths  
  - `outpath`: where to write the final labels PKL

> **Note:** `main.py` imports `utils` for `parse_config_file` and `load_fr_model`. Ensure `utils.py` (or an equivalent module) is available on `PYTHONPATH` and provides those functions.

---

## How it works (short paper-aligned overview)

1. **Flip:** Compare the FR embedding of an image with its horizontally flipped version via cosine similarity. Higher similarity ⇒ higher quality.  
2. **Noise:** Mix the image with small Gaussian noise at multiple α values, embed, and average cosine similarities vs. the original.  
3. **Occlusion:** Mask non-overlapping grid squares across the image, embed each occluded sample, average cosine similarities vs. original.  
4. **Combine:** Average the three component scores into a single pseudo-quality label per image.

The code mirrors the above exactly in `flip_component.py`, `noise_component.py`, and `occlusion_component.py`, and aggregates them in `main.py`.

---


## Configuration

Example (`baselineFIQA/baseline_FIQA_config.yaml`):

```yaml
face_recognition_config: ./path/to/fr_model_config.yaml

dataset:
  path: /path/to/your/images
  batch_size: 64
  mpx: 14          # occlusion square size (pixels)
  image_size: 112  # masks sized for your FR input size
  alphas: [0.001, 0.002, 0.005]  # noise strengths

outpath: ./labels/aux_fiqa_labels.pkl
```

The `face_recognition_config` must be compatible with `utils.load_fr_model(...)`, which should return `(fr_model, transform)`.

---

## Running

From the repo root:

```bash
python3 ./baselineFIQA/main.py -c ./baselineFIQA/baseline_FIQA_config.yaml
```

This will:
- walk `dataset.path` to collect images,  
- compute flip/occlusion/noise scores,  
- write temporary `flip_scores.pkl`, `occlusion_scores.pkl`, `noise_scores.pkl`,  
- and finally write the combined labels to `outpath`.

---

## File-by-file usage notes

- **Using the components directly**
  - `flip_component.generate_flip_scores(fr_model, transform, loc, batch_size, torch.nn.CosineSimilarity()) → dict`
  - `noise_component.generate_noise_scores(fr_model, transform, loc, batch_size, torch.nn.CosineSimilarity(), alphas) → dict`
  - `occlusion_component.generate_occlusion_scores(fr_model, transform, loc, batch_size, torch.nn.CosineSimilarity(), mpx, image_size) → dict`

- **Datasets**
  - All three components subclass `Glint360KSubset_Base(trans, loc)`, which recursively gathers file paths under `loc`; `__getitem__` loads/transforms images as needed.

---
