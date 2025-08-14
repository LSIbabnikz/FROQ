# Observer: Layer Utility–Based Quality Assessment

This module implements the *Observer* method for face image quality estimation.  
It works by **finding the most informative intermediate layers** of a face recognition (FR) model for predicting image quality, using pseudo-quality labels (e.g., from FIQA). The method proceeds in two stages:

1. **Initialization** – Evaluate each layer’s correlation with pseudo-quality labels, then greedily build a *final layer set* that maximizes utility for quality assessment.
2. **Inference** – Using the chosen layer set, hook into the FR model during forward passes, compute feature norms per layer, and average them to get a quality score per image.

---

## Repository structure

> All paths below assume the `observer/` subfolder.

- **`initialization.py`** – Runs the *initialization stage*.  
  - Loads FR model and transform.  
  - Scans the calibration dataset and evaluates the Spearman rank correlation between each layer’s feature norms and pseudo-quality labels.  
  - Greedily constructs the final layer set for quality assessment.  
  - Saves `(single_layer_utilities, pseudo_quality_labels_ordered, layer_set, intermediate_set_correlations)` to a pickle file for later inference.

- **`inference.py`** – Runs the *inference stage*.  
  - Loads the FR model, the dataset, and the saved initialization pickle.  
  - Hooks into the selected layers, computes average feature norms, and produces a per-image quality score dictionary.  
  - Saves the results to the configured `outpath`.

- **`dataset.py`** – Contains two dataset wrappers:  
  - `Glint360KSubset`: Recursively collects all image files under a directory and returns `(image_path, transformed_image)`【42†source】.  
  - `DummyDataset`: Same behavior, but used in inference for arbitrary datasets.

- **`initialization_config.yaml`** – Example config for initialization.  
- **`inference_config.yaml`** – Example config for inference.

---

## Configuration

### Initialization config (`initialization_config.yaml`)

```yaml
face_recognition_config: ./path/to/fr_model_config.yaml

dataset:
  path: /path/to/calibration/images
  batch_size: 64

label_location: ./path/to/pseudo_quality_labels.pkl
outpath: ./observer_data.pkl
```

- **`face_recognition_config`**: Path to config understood by `utils.load_fr_model(...)`.  
- **`dataset.path`**: Calibration image root folder.  
- **`label_location`**: Pickle file with `{image_path: pseudo_quality_score}` mapping.  
- **`outpath`**: Where to save the observer initialization data.

---

### Inference config (`inference_config.yaml`)

```yaml
face_recognition_config: ./path/to/fr_model_config.yaml

target_dataset: /path/to/target/images
batch_size: 64

observer_data: ./observer_data.pkl
outpath: ./observer_quality_scores.pkl
```

- **`target_dataset`**: Image folder for inference.  
- **`observer_data`**: Path to the pickle from the initialization stage.  
- **`outpath`**: Output pickle with `{image_path: quality_score}` mapping.

---

## Running

1. **Initialization** (select the most useful layers):

```bash
python3 ./observer/initialization.py -c ./observer/initialization_config.yaml
```

2. **Inference** (compute quality scores for a dataset):

```bash
python3 ./observer/inference.py -c ./observer/inference_config.yaml
```
