import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, unzip_file
from src.utils.data_mgmt import load_mnist_data
import random
import numpy as np

STAGE = "GET_DATA"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def main(config_path):
    config = read_yaml(config_path)

    train_images, train_labels, test_images, test_labels = load_mnist_data()
    logging.info("Loaded train_images, train_labels, test_images, test_labels")

    local_dir = config["data"]["local_dir"]
    create_directories([local_dir])

    train_images_dir = config["data"]["train_images_dir"]
    train_labels_dir = config["data"]["train_labels_dir"]
    test_images_dir = config["data"]["test_images_dir"]
    test_labels_dir = config["data"]["test_labels_dir"]

    # Debugging: Check directory paths
    logging.info(f"train_images_dir: {train_images_dir}")
    logging.info(f"train_labels_dir: {train_labels_dir}")
    logging.info(f"test_images_dir: {test_images_dir}")
    logging.info(f"test_labels_dir: {test_labels_dir}")

    create_directories([train_images_dir, train_labels_dir, test_images_dir, test_labels_dir])

    # Save train images
    np.save(os.path.join(train_images_dir, "train_images.npy"), train_images)
    # Save train labels
    np.save(os.path.join(train_labels_dir, "train_labels.npy"), train_labels)
    # Save test images
    np.save(os.path.join(test_images_dir, "test_images.npy"), test_images)
    # Save test labels
    np.save(os.path.join(test_labels_dir, "test_labels.npy"), test_labels)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed! <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
