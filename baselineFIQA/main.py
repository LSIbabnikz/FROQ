

import sys
sys.path.append(".")

import pickle
import argparse

from flip_component import *
from noise_component import *
from occlusion_component import *

from utils import *  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    cfg = parse_config_file(args.config)

    # Get FR model from config
    face_recognition_config = parse_config_file(cfg["face_recognition_config"])
    fr_model, transform = load_fr_model(face_recognition_config)

    dataset_config = cfg["dataset"]

    cossim = torch.nn.CosineSimilarity()

    # Run the three perturbation components
    flip_scores = generate_flip_scores(
        fr_model, transform, dataset_config["path"], dataset_config["batch_size"], cossim)
    occlusion_scores = generate_occlusion_scores(
        fr_model, transform, dataset_config["path"], dataset_config["batch_size"], cossim,
        dataset_config["mpx"], dataset_config["image_size"])
    noise_scores = generate_noise_scores(
        fr_model, transform, dataset_config["path"], dataset_config["batch_size"], cossim,
        dataset_config["alphas"])

    # Combine the components
    combined_score = {}
    for image_path in flip_scores.keys():
        combined_score[image_path] = \
            1./3. * (flip_scores[image_path] + noise_scores[image_path] + occlusion_scores[image_path])

    # Used for debugging (uncomment if needed)
    # with open("./baselineFIQA/flip_scores.pkl", "wb") as pkl_out:
    #     pickle.dump(flip_scores, pkl_out)
    # with open("./baselineFIQA/occlusion_scores.pkl", "wb") as pkl_out:
    #     pickle.dump(occlusion_scores, pkl_out)
    # with open("./baselineFIQA/noise_scores.pkl", "wb") as pkl_out:
    #     pickle.dump(noise_scores, pkl_out)

    # Save pseudo-quality labels
    with open(cfg["outpath"], "wb") as pkl_out:
        pickle.dump(combined_score, pkl_out)
