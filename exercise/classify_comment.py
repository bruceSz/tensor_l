import tensorflow as tf
import numpy as np

import random
import pickle
from collections import Counter

pos_txt = "../../data/tf_dat/pos.txt"
neg_txt = "../../data/tf_dat/neg.txt"

def create_lexicon(pos_file,neg_file):
    lex = []
    def process_file(f):
        with open(pos_file,'r') as f:
            lex = []
            lines = f.readlines()


def main():
    pass


if __name__ == "__main__":
    main()


