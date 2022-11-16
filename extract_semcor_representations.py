'''This script extracts c2v/bert representations for semcor data.
For à la carte embeddings, see readme'''


import numpy as np
import os
from context2vec.common.model_reader import ModelReader
import re
import pandas as pd
import argparse
import pdb
from transformers import BertModel, BertTokenizer, BertConfig
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dump_vectors(vector_dict, vectorfile):
  '''dumps embeddings to .txt
  Args:
    generator: (gram, vector) generator; vector can also be a scalar
    vectorfile: .txt file
  Returns:
    None
  '''
  with open(vectorfile, 'w') as f:
    for gram, vector in vector_dict.items():
      numstr = ' '.join(map(str, vector.tolist())) if vector.shape else str(vector)
      f.write(gram + ' ' + numstr + '\n')

def smart_tokenization(sentence, tokenizer, maxlen):
    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    map_ori_to_bert = []
    tok_sent = [cls_token]
    incomplete = False
    for orig_token in sentence:
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) # tokenize
        ##### check if adding this token will result in >= maxlen (=, because [SEP]). If so, stop
        if len(tok_sent) + len(bert_token) >= maxlen:
            incomplete = True
            break
        tok_sent.extend(bert_token) # add to my new tokens
        if len(bert_token) > 1: # if the new token has been 'wordpieced'
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1) # list of new positions of the target word in the new tokenization
        map_ori_to_bert.append(current_tokens_bert_idx)

    tok_sent.append(sep_token)

    return tok_sent, map_ori_to_bert, incomplete


#################### Functions for c2v

class ParseException(Exception):
    def __init__(self, str):
        super(ParseException, self).__init__(str)

target_exp = re.compile('\[.*\]')

def parse_input(line):
    #sent = line.strip().split()
    sent = line
    target_pos = None
    for i, word in enumerate(sent):
        if target_exp.match(word) != None:
            target_pos = i
            if word == '[]':
                word = None
            else:
                word = word[1:-1]
            sent[i] = word
    return sent, target_pos



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', default="semcor_representations/")
    parser.add_argument('--vector_type', default="bert")
    parser.add_argument('--c2v_filename', default="context2vec/models/context2vec.ukwac.model.params", help="this is the .params file of a context2vec model.")
    args = parser.parse_args()

    if args.vector_type == "c2v":
        model_reader = ModelReader(args.c2v_filename)
        w = model_reader.w
        word2index = model_reader.word2index
        index2word = model_reader.index2word
        model = model_reader.model
    elif args.vector_type == "bert":
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        model = BertModel.from_pretrained("bert-base-uncased", config=config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    diri = args.corpus_dir
    if diri[-1] != "/":
        diri += "/"
    print("model loaded")
    j = -1
    for target in os.listdir(diri): # here target is a lemma
        if os.path.isdir(diri + target):
            print(target)
            j+=1
            if j % 50 == 0:
                print("*************Done", j, "targets*************")

            infodf = pd.read_csv(diri + target+"/" + target + "_info.csv", sep="\t")
            indices = infodf["index"]
            positions = infodf['position']
            corpusfile = diri + target + "/" + target + "_sentences.txt"
            with open(corpusfile) as f:
                sentences = [l.strip().split() for l in f]

            if args.vector_type == "c2v":
                context_vecs = []
                for sentence, p in zip(sentences, positions):
                    sentcopy = sentence[:]
                    sentcopy[p] = '[]'
                    sent, target_position = parse_input(sentcopy)
                    context_v = model.context2vec(sent, target_position)
                    context_v = context_v / np.sqrt((context_v * context_v).sum())
                    context_vecs.append(context_v)
                print("read vectors")
                #### save the vectors
                dump_vectors({target.split("_")[0] + "-" + str(i): cv for i, cv in enumerate(context_vecs)}, diri + target + "/" + target + "_c2v-all.txt")
                #### make the averages for different Xs
                for X in [3, 5, 10, 20, 25]:
                    classes = infodf["X" + str(X)]
                    indices1 = np.array([i for i in range(len(classes)) if classes[i] == 1])
                    indices2 = np.array([i for i in range(len(classes)) if classes[i] == 2])
                    vector1 = {target.split("_")[0]: np.average(np.array(context_vecs)[indices1], axis=0)}
                    vector2 = {target.split("_")[0]: np.average(np.array(context_vecs)[indices2], axis=0)}
                    outtext1 = "part1_X" + str(X)
                    outtext2 = "part2_X" + str(X)
                    root1 = diri + target + "/" + target + "_" + outtext1 + "_c2v-contextvectors.txt"
                    root2 = diri + target + "/" + target + "_" + outtext2 + "_c2v-contextvectors.txt"

                    dump_vectors(vector1, root1)
                    dump_vectors(vector2, root2)

            elif args.vector_type == "bert":
                # first, extract all vectors, for all sentences. At the end, we average them.
                vecs_by_layer = dict() # keys are layers and values are lists of len 50 containing vectors for the 50 instances).
                for sentence, p in zip(sentences, positions):
                    bert_tokens, map_ori_to_bert, incomplete = smart_tokenization(sentence, tokenizer,
                                                                                  maxlen=model.config.max_position_embeddings)
                    # in this case p is an int
                    bert_target_idcs = map_ori_to_bert[p]
                    model.eval()
                    model.to(device)
                    with torch.no_grad():
                        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(bert_tokens)]).to(device)
                        inputs = {'input_ids': input_ids}
                        outputs = model(**inputs)
                        hidden_states = outputs[2]
                        bpereps_for_this_instance = dict()
                        for occurrence_idx in bert_target_idcs:
                            w = bert_tokens[occurrence_idx]
                            for layer in range(len(hidden_states)):
                                if layer not in bpereps_for_this_instance:
                                    bpereps_for_this_instance[layer] = []
                                bpereps_for_this_instance[layer].append((w, hidden_states[layer][0][
                                    occurrence_idx].cpu()))  # 0 is for the batch
                        # if it consists of multiple subwords, average those
                        for layer in bpereps_for_this_instance:
                            if layer not in vecs_by_layer:
                                vecs_by_layer[layer] = []
                            vecs_by_layer[layer].append(np.average(np.array([rep.numpy() for w, rep in bpereps_for_this_instance[layer]]), axis=0))
                for layer in vecs_by_layer:
                    for X in [3, 5, 10, 20, 25]:
                        classes = infodf["X" + str(X)]
                        indices1 = np.array([i for i in range(len(classes)) if classes[i] == 1])
                        indices2 = np.array([i for i in range(len(classes)) if classes[i] == 2])
                        vector1 = {target.split("_")[0]: np.average(np.array(vecs_by_layer[layer])[indices1], axis=0)}
                        vector2 = {target.split("_")[0]: np.average(np.array(vecs_by_layer[layer])[indices2], axis=0)}
                        outtext1 = "part1_X" + str(X)
                        outtext2 = "part2_X" + str(X)
                        root1 = diri + target + "/" + target + "_" + outtext1 + "_bert-" + str(layer) + ".txt"
                        root2 = diri + target + "/" + target + "_" + outtext2 + "_bert-" + str(layer) + ".txt"

                        dump_vectors(vector1, root1)
                        dump_vectors(vector2, root2)


