'''With this script we obtain bert embeddings for stance data'''

import numpy as np
import os
import argparse
import pdb
import json
from transformers import BertModel, BertTokenizer, BertConfig
import torch
from extract_semcor_representations import smart_tokenization
from utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="semeval2016", help="can be semeval2016, covid19, pstance, 30k")
    args = parser.parse_args()

    out_dir = "stance_split_data/"  + args.dataset_name + "/"
    middle_dirs = [folder + "/" for folder in os.listdir(out_dir)] # target dirs

    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model = BertModel.from_pretrained("bert-base-uncased", config=config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("model loaded")
    for target_dir in middle_dirs:
        diri = out_dir + target_dir
        for subset in os.listdir(diri):
            print(subset)
            for lemma in os.listdir(diri + subset + "/"):
                print(lemma)
                fulldir = diri + subset + "/" + lemma + "/"
                if os.path.isdir(fulldir):
                    with open(fulldir + "info.txt") as f:
                        infos = json.load(f)
                    with open(fulldir + "sentences.txt") as f:
                        sentences = [l.strip().split() for l in f]
                    instance_reps_per_layer = dict()
                    for sentence, info in zip(sentences, infos):
                        bert_tokens, map_ori_to_bert, incomplete = smart_tokenization(sentence, tokenizer, maxlen=model.config.max_position_embeddings)
                        if incomplete:
                            print("Incomplete sentence (the target token is at a position that exceeds the max length)")
                            pdb.set_trace() # didn't happen with this data but a decision should be taken if it happens
                        for position in info["lemma_positions"][lemma]:
                            bert_target_idcs = map_ori_to_bert[position]
                            model.eval()
                            model.to(device)
                            # Obtain contextualized representations
                            with torch.no_grad():
                                input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(bert_tokens)]).to(device)
                                inputs = {'input_ids': input_ids}
                                outputs = model(**inputs)
                                hidden_states = outputs[2]
                                bpereps_for_this_instance = dict()  # keys will be the layers
                                for occurrence_idx in bert_target_idcs:
                                    w = bert_tokens[occurrence_idx]
                                    for layer in range(len(hidden_states)):  # all layers
                                        if layer not in bpereps_for_this_instance:
                                            bpereps_for_this_instance[layer] = []
                                        bpereps_for_this_instance[layer].append((w, hidden_states[layer][0][occurrence_idx].cpu())) # 0 is for the batch
                                # if it consists of multiple subwords, average those
                                for layer in bpereps_for_this_instance:
                                    if layer not in instance_reps_per_layer:
                                        instance_reps_per_layer[layer] = []
                                    instance_reps_per_layer[layer].append( np.average(np.array([rep.numpy() for w, rep in bpereps_for_this_instance[layer]]), axis=0) )
                    # now save them separately by layer
                    for layer in instance_reps_per_layer:
                        vectors_to_dump = {lemma.split("_")[0] : np.average(instance_reps_per_layer[layer], axis=0)}
                        dump_vectors(vectors_to_dump, fulldir + "bert_" + str(layer) + ".txt")


