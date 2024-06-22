import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="EleutherAI/pythia-6.9b", help="the model to attack")
        self.parser.add_argument('--ref_model', type=str, default="EleutherAI/pythia-160m")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--dataset', type=str, help="dataset name")
        self.parser.add_argument('--sub_dataset', type=int, default=128, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--num_shots', type=str, default="12", help="number of shots to evaluate.")
        self.parser.add_argument('--pass_window', type=bool, default=True, help="whether to pass the window to the model.")
        self.parser.add_argument("--synehtic_prefix", type=bool, default=False, help="whether to use synehtic prefix.")
        self.parser.add_argument("--api_key_path", type=str, default=None, help="path to the api key file for OpenAI API if using synehtic prefix.")




