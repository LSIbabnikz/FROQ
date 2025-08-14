
import sys
sys.path.append(".")

import pickle
import argparse

import torch

from tqdm import tqdm

from utils import *
from observer.dataset import DummyDataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    cfg = parse_config_file(args.config)

    # Get FR model from config
    face_recognition_config = parse_config_file(cfg["face_recognition_config"])
    fr_model, transform = load_fr_model(face_recognition_config)
    fr_model.cuda().eval()

    # Get dataset loader
    dataset = DummyDataset(transform, cfg["target_dataset"]) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch_size"])

    # Determine layers to observe from initialization data
    model_layers = list(fr_model.named_modules())[1:]
    with open(cfg["observer_data"], "rb") as pkl_in:
        (single_layer_utilities,
        _,
        layer_set, 
        intermediate_set_correlations) = pickle.load(pkl_in) 

    # Take layers up until max utility of set
    index_of_best_set = intermediate_set_correlations.index(max(intermediate_set_correlations))
    top_set_layer_indices = layer_set[:index_of_best_set + 1]
    # Order layers according to utility and take the determined indices
    layer_ordering = torch.argsort(torch.tensor(single_layer_utilities), descending=True)
    final_set_layers = layer_ordering[top_set_layer_indices]

    # Dictionary stores intermediate representations
    layer_outputs = {}
    # Hook function to apply to forward hooks
    def hook_fn(layer_name):
        def hook(module, input, output):
            layer_outputs[layer_name] = output.detach().clone()
        return hook

    # Set hooks for layers in final set
    hooks = []
    for layer in final_set_layers:
        hook = model_layers[layer][1].register_forward_hook(hook_fn(str(layer)))
        hooks.append(hook)

    full_quality_scores = {}
    with torch.no_grad():
        for (name_batch, img_batch) in tqdm(dataloader, desc=' Evaluating quality of dataset : '):

            _ = fr_model(img_batch.cuda()).detach()

            batch_quality_scores = torch.zeros((len(name_batch),))
            for layer in final_set_layers:
                norms = torch.linalg.norm(layer_outputs[str(layer)].reshape(len(name_batch), -1), dim=1).detach().cpu()
                batch_quality_scores += norms

            batch_quality_scores = batch_quality_scores / len(final_set_layers)

            full_quality_scores.update(
                dict(zip(
                    name_batch, 
                    batch_quality_scores.squeeze().numpy().tolist()
                    ))
                )

    with open(cfg["outpath"], "wb") as pkl_out:
        pickle.dump(full_quality_scores, pkl_out)
