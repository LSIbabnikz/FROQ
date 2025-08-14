
import sys
sys.path.append(".")

import os
import pickle
import argparse
from typing import Tuple

from torch.utils.data import Dataset

from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr

from utils import *


class Glint360KSubset(Dataset):

    def __init__(
            self, 
            trans,
            loc):   
        self.items = []
        for (dir, subdirs, files) in os.walk(loc):
            self.items.extend(list(filter(lambda x: x.endswith("jpg"), map(lambda x: os.path.join(dir, x), files))))
        self.trans = trans

    def __getitem__(self, x):
        image_loc = self.items[x]
        image = self.trans(Image.open(image_loc).convert("RGB"))

        return image_loc, image
    
    def __len__(self):
        return len(self.items)
  

def define_dict_and_hook_fn():
    # Dictionary stores intermediate representations
    layer_outputs = {}
    # Hook function to apply to forward hooks
    def hook_fn(layer_name):
        def hook(module, input, output):
            layer_outputs[layer_name] = output.detach().clone()
        return hook

    return layer_outputs, hook_fn


def calc_corr(
        layer_quality_scores: list, 
        set_of_layers: list, 
        pseudo_quality_labels: list
        ) -> float:
    """Helper function calculates utility of given set, using individial layers scores and pseudo qualities.

    Args:
        layer_quality_scores (list): Quality scores of individual layers over the calibration set.
        set_of_layers (list): Set of layers to include in utility calculation.
        pseudo_quality_labels (list): Quality labels for correlation coefficient calcualtion.

    Returns:
        float: The utility of the given set of layers for the taks of quality assessment.
    """

    set_quality_scores = None
    for idx in set_of_layers:
        if set_quality_scores is None:
            set_quality_scores = layer_quality_scores[idx]
        else:
            set_quality_scores = [el1 + el2 for el1, el2 in zip(set_quality_scores, layer_quality_scores[idx])]
    set_quality_scores = [el/len(set_of_layers) for el in set_quality_scores]

    set_utility = float(spearmanr(np.array(set_quality_scores), np.array(pseudo_quality_labels))[0])

    return set_utility



@torch.no_grad()
def evaluate_single_layers(
    fr_model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    label_path: str,
    layer_batch_size: int = 20) -> Tuple[list, list]:
    """Evaluates the utility of individual model layers for the task of quality assessment.

        Args:
            fr_model (torch.nn.Module): The FR model, whose layers should be analyzed.
            dataloader (torch.utils.data.DataLoader): Dataloader of calibration images.
            label_path (str): Path to the pseudo quality labels of calibration images.
            layer_batch_size (int, optional): Number of layers to process in parallel. Defaults to 20.

    Returns:
        Tuple[list, list]: Returns a tuple of lists containing the scores of individual layers and reordered list of images.
    """

    # Get all layers from given FR model
    model_layers = list(fr_model.named_modules())[1:]
    layer_count = len(model_layers)
    
    # Load the generated pseudo-quality labels
    with open(label_path, "rb") as pkl_in:
        pseudo_quality_label_data = pickle.load(pkl_in)

    # Reorder data in order for proper comparison using Spearman rank correlation
    desired_reorder = []
    for (name_batch, _) in dataloader:
        desired_reorder.extend(name_batch)
    pseudo_quality_labels_ordered = [pseudo_quality_label_data[dn] for dn in desired_reorder]

    single_layer_utilities = []

    # Dictionary stores intermediate representations
    layer_outputs = {}
    # Hook function to apply to forward hooks
    def hook_fn(layer_name):
        def hook(module, input, output):
            layer_outputs[layer_name] = output.detach().clone()
        return hook

    # Process layers in batches
    for i in tqdm(range(0, layer_count, layer_batch_size), desc=' Processing individual layers : '):
        
        layer_qualities = [[] for _ in range(i, min(i+layer_batch_size, layer_count))]

        # Apply forward hooks to all layers in current batch
        hooks = []
        for jj in range(i, min(i+layer_batch_size, layer_count)):
            hook = model_layers[jj][1].register_forward_hook(hook_fn(str(jj)))
            hooks.append(hook)

        # Feed calibration set through model
        for (name_batch, img_batch) in dataloader:

            _ = fr_model(img_batch.cuda()).detach()

            # For each layer compute the pseudo quality by applying the frobenious norm
            for jj in range(len(layer_qualities)):
                if str(i+jj) in layer_outputs:
                    layer_qualities[jj].extend(
                        torch.linalg.norm(
                            layer_outputs[str(i+jj)].reshape(len(name_batch), -1), 
                            dim=1).detach().cpu().numpy().tolist())
                else:
                    layer_qualities[jj].extend(torch.randn((len(name_batch),)).detach().cpu().numpy().tolist())
        
        # Remove hooks from current batch of layers
        for hk in hooks:    
            hk.remove()
    
        # Determine the rank coefficient between qualities from individual layers and pseudo quality labels
        for jj in range(len(layer_qualities)):

            layer_correlation_coefficient = float(
                spearmanr(
                    np.array(layer_qualities[jj]), 
                    np.array(pseudo_quality_labels_ordered)
                    )[0])
            single_layer_utilities.append(layer_correlation_coefficient)

        layer_outputs = {}

    return single_layer_utilities, pseudo_quality_labels_ordered


@torch.no_grad()
def build_final_layer_set(
    fr_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    single_layer_utilities: list,
    pseudo_quality_labels_ordered: list,
    keep_top_n_layers: int = 10
    ) -> Tuple[list, list]: 
    """Uses a greedy search algorithm to construct the set of layers which maximize the utility for quality assessment.

    Args:
        fr_model (torch.nn.Module): The FR model, whose layers should be analyzed.
        dataloader (torch.utils.data.DataLoader): Dataloader of calibration images.
        single_layer_utilities (list): Utility values of individual layers of fr_model, produced by __evaluate_single_layers()__
        pseudo_quality_labels_ordered (list): Ordered values of pseudo quality labels, producved by __evaluate_single_layers()__
        keep_top_n_layers (int, optional): Number of layers to consider when constructing the final set. Defaults to 10.

    Returns:
        Tuple[list, list]: Returns the ordered indices of layers added to the set and the respective utility scores.
    """

    # Determine the top n layers
    top_n_layers = torch.argsort(
            torch.tensor(single_layer_utilities), descending=True
        ).numpy().tolist()[:keep_top_n_layers]
    model_layers = list(fr_model.named_modules())[1:]

    # Dictionary stores intermediate representations
    layer_outputs = {}
    # Hook function to apply to forward hooks
    def hook_fn(layer_name):
        def hook(module, input, output):
            layer_outputs[layer_name] = output.detach().clone()
        return hook

    # Apply hooks to all top-n layers
    hooks = []
    for layer_name in top_n_layers:
        hook = model_layers[layer_name][1].register_forward_hook(hook_fn(str(layer_name)))
        hooks.append(hook)

    # Get quality labels from top-n layers
    top_layer_qualities = [[] for _ in range(keep_top_n_layers)]
    for (name_batch, image_batch) in tqdm(dataloader, desc=' Scoring top-n layers : '):

        _ = fr_model(image_batch.cuda()).detach()

        for i in range(len(top_n_layers)):
            top_layer_qualities[i].extend(
                torch.linalg.norm(
                        layer_outputs[str(top_n_layers[i])].reshape(len(name_batch), -1), 
                        dim=1
                    ).detach().cpu().numpy().tolist())

    # Run the greedy search algorithm over the top-n layers
    # Initialize the set using the best layer then at each step add the layer that minimizes the joint coefficient
    print(f" Building final set of layers: ")
    layer_set = [0]
    intermediate_set_correlations = []
    intermediate_set_correlations.append(
        calc_corr(top_layer_qualities, layer_set, pseudo_quality_labels_ordered)
        )
    available_top_layers = list(range(keep_top_n_layers))
    available_top_layers.remove(0)

    print(f" \t => Added: {layer_set[0]} - ({layer_set}/{intermediate_set_correlations[-1]})")
    for i in range(1, keep_top_n_layers):
        # Test each still available layer and chose the best one
        potential = []
        for available_layer in available_top_layers:
            layer_set_plus_available = layer_set.copy()
            layer_set_plus_available.append(available_layer)
            potential.append((available_layer, 
                              calc_corr(top_layer_qualities, layer_set_plus_available, pseudo_quality_labels_ordered)))
        potential = sorted(potential, key=lambda x: x[1], reverse=True)

        layer_set.append(potential[0][0])
        intermediate_set_correlations.append(potential[0][1])
        available_top_layers.remove(potential[0][0])
        print(f" \t => Added: {layer_set[-1]} - ({layer_set}/{intermediate_set_correlations[-1]})")

    return layer_set, intermediate_set_correlations



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    cfg = parse_config_file(args.config)

    # Get FR model from config
    face_recognition_config = parse_config_file(cfg["face_recognition_config"])
    fr_model, transform = load_fr_model(face_recognition_config)
    fr_model.cuda().eval()

    # Get dataset and dataloader
    dataset_config = cfg["dataset"]
    dataset = Glint360KSubset(transform, dataset_config["path"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config["batch_size"])

    # Get individual layer scores according to pseudo quality labels
    single_layer_utilities, pseudo_quality_labels_ordered = \
        evaluate_single_layers(fr_model, dataloader, cfg["label_location"])

    # Combine best individual layers into final set 
    layer_set, intermediate_set_correlations = \
        build_final_layer_set(fr_model, dataloader, single_layer_utilities, 
                                pseudo_quality_labels_ordered)
    
    # Save the observer initialization data
    with open(cfg["outpath"], "wb") as pkl_out:
        pickle.dump((single_layer_utilities,
                     pseudo_quality_labels_ordered,
                     layer_set, 
                     intermediate_set_correlations), pkl_out)