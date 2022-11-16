

import pandas as pd
import random
import os
from utils import *
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
import argparse
import sys
from sklearn.metrics.pairwise import cosine_distances
import pdb
from collections import OrderedDict


from string import punctuation

random.seed(0)

def load_gold_standard(semcor_dir="semcor_representations/"):
    fn = semcor_dir + "global_info.csv"
    gs = pd.read_csv(fn, sep="\t")
    return gs

def calculate_changes(approach, gs, lemma_infos, method="cosine"):
    Xs = [3, 5, 10, 20, 25]
    target_words = []
    changes = {X: [] for X in Xs}
    for targetword, pos in zip(gs['lemma'], gs['pos']):
        targetword = targetword + "_" + pos
        target_words.append(targetword)
        for X in Xs:
            change = cosine(approach.vectors[X][targetword]['part1'], approach.vectors[X][targetword]['part2'])
            changes[X].append(change)

    try:
        for X in changes:
            gs[method + "_X" + str(X)] = changes[X]
            approach.results[method + "_X" + str(X)] = changes[X]
    except ValueError:
        pdb.set_trace()
    if not approach.lemmas:
        approach.lemmas = target_words


def save_changes(gs, out_dir="semcor_predictions/", save_gs=False):
    df = pd.DataFrame()
    sensestr = "s_"
    method = "cosine"
    for col in gs:
        if not save_gs: # exclude gold standard from file
            if col.startswith(method) or col in ['lemma', 'pos']:
                df[col] = gs[col]
        elif save_gs == "only": # only save gs
            if col in ['lemma','pos'] or col.startswith(sensestr + "JSD"): # we are saving only gold standard, separately
                df[col] = gs[col]
        elif save_gs == "both":
            if col.startswith(method) or col in ['lemma', 'pos'] or col.startswith(sensestr + "JSD"):  # include gold standard to make things easier...
                df[col] = gs[col]
    if save_gs != "only":
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df.to_csv(out_dir + method + ".csv", sep="\t", index=False)


    if save_gs == "only":
        if not os.path.exists("/".join(out_dir.split("/")[:-1])):
            os.makedirs("/".join(out_dir.split("/")[:-1]))
        df.to_csv(out_dir + ".csv", sep="\t", index=False)


class Approach():
    def __init__(self, vector_type, vector_size, layer):
        if vector_type == "bert":
            self.vector_type = "bert-"+ str(layer)
        else:
            self.vector_type = vector_type
        self.vector_size = vector_size if vector_type == "ALC" else "600"
        self.vectors = dict()
        self.individual_vectors = dict()
        self.results = dict()
        self.lemmas = []
        self.name = self.vector_type + "_" + self.vector_size + "_senses"

    def load_new_vectors(self, num_sentences, paths):
        num_sentences = str(num_sentences)
        vectors = dict()
        if self.vector_type == "ALC":
            finaltxt = "_alacarte"
        elif self.vector_type == "c2v":
            finaltxt = ""
            self.vector_size = "c2v-contextvectors"
        elif "bert" in self.vector_type:
            finaltxt = ""
            self.vector_size = self.vector_type
        else:
            print("NOT IMPLEMENTED")

        for path in paths:
            targetfolder = path.split("/")[-1]
            vectors[targetfolder] = dict()
            corpus1_vecs = OrderedDict(load_vectors(path + "/" + targetfolder + "_part1_X" +
                                                    str(num_sentences) + "_" + self.vector_size + finaltxt + ".txt"))
            corpus2_vecs = OrderedDict(load_vectors(path + "/" + targetfolder + "_part2_X" +
                                                    str(num_sentences) + "_" + self.vector_size + finaltxt + ".txt"))
            vectors[targetfolder]['part1'] = list(corpus1_vecs.values())[0]
            vectors[targetfolder]['part2'] = list(corpus2_vecs.values())[0]

        self.vectors[X] = vectors

        if self.vector_type == "c2v": # load the individual vectors too
            for path in paths:
                targetfolder = path.split("/")[-1]
                vecs = OrderedDict(load_vectors(path + "/" + targetfolder + "_c2v-all.txt"))
                self.individual_vectors[targetfolder] = list(vecs.values())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--vector_type', default="bert", help='c2v, ALC or bert')
    parser.add_argument('--vector_size', default="840B.300d", help='if vector type is ALC, give size of glove '
                                                                 'embeddings used. 6B.300d, 840B.300d')
    args = parser.parse_args()

    semcor_dir = "semcor_representations/"

    # Load the gold standard data
    gs = load_gold_standard(semcor_dir=semcor_dir)
    paths_to_vectors = find_semcor_paths(semcor_dir=semcor_dir)

    lemma_infos = dict()
    for path in paths_to_vectors:
        targetfolder = path.split("/")[-1]
        lemma_infos[targetfolder] = pd.read_csv(path + "/" + targetfolder + "_info.csv", sep="\t")

    layers = range(0,13) if args.vector_type == "bert" else [0]

    for layer in layers:
        print(layer)
        approach = Approach(vector_type=args.vector_type, vector_size=args.vector_size, layer=layer)

        # Load embeddings to be evaluated
        for X in [3, 5, 10, 20, 25]:
            approach.load_new_vectors(num_sentences=X, paths=paths_to_vectors)

        # Calculate a change score
        calculate_changes(approach, gs, lemma_infos=lemma_infos)

        if args.vector_type == "ALC":
            vector_str = "ALC-" + args.vector_size
        else:
            vector_str = approach.vector_type

        out_dir = "semcor_predictions/"

        out_dir_vec = out_dir + vector_str + "/"
        save_changes(gs, out_dir=out_dir_vec)

        # save also the gs separately
        out_dir_gs =out_dir + "gold_standard-senses"
        if not os.path.exists(out_dir_gs):
            save_changes(gs, out_dir=out_dir_gs, save_gs="only")



