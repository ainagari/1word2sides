
import argparse
from utils import *
import os
import json
from scipy.spatial.distance import cosine
from collections import defaultdict
from scipy.stats import shapiro, ttest_rel, wilcoxon, chisquare
from collections import OrderedDict



def load_vectors_and_info_stance(in_dir, layer):
    '''
    This function loads c2v or bert vectors previously extracted for one target (specified in in_dir) for the stance datasets. It also loads some extra info.
    Args:
        in_dir: path to the vectors for the a specific target
        layer: bert layer, int
    Returns:
        two dicts with the same structure, one containing data (tokens, lemmas...) and the other one vectors.
        keys are subset names (e.g. FAVOR-1) and the value is another dict with words as keys
    '''
    # load embeddings and information (we will want to use frequency information for weighting)
    data, vectors = dict(), dict()
    for subset in os.listdir(in_dir): # do not load vectors for NONE stances (if there are any)
        if "NONE" in subset:
            continue
        middir = in_dir + subset + "/"
        if not os.path.isdir(middir): # then it is a file
            continue
        data[subset], vectors[subset] = dict(), dict()
        for word in os.listdir(middir):
            fulldir = middir + word + "/"
            if os.path.isdir(fulldir):
                with open(fulldir + "info.txt") as f:
                    data[subset][word] = json.load(f)
                vectors[subset][word] = list(OrderedDict(load_vectors(fulldir + "bert_" + str(layer) + ".txt")).values())[0]
    return data, vectors



def calculate_similarities_stance(data, vectors, groupings):
    '''Calculate within-WORD similarities for every comparison in this data'''
    similarities = dict()
    for groupname in groupings:
        subset1, subset2 = groupings[groupname] # e.g favor1, against2
        similarities[groupname] = dict()
        for lemma in data[subset1]:
            if lemma in data[subset2]:
                change = cosine(vectors[subset1][lemma], vectors[subset2][lemma]) # distance
                similarities[groupname][lemma] = 1 - change # similarity
    return similarities


def load_tfidf(in_dir, target):
    '''Load previously calculated tfidf values'''
    fn = in_dir + target + "/tfidf.tsv"
    with open(fn) as f:
        tfidf_dict = {line.split("\t")[0]:float(line.split("\t")[1]) for line in f.readlines()}
    return tfidf_dict


def calculate_comparison_similarity(similarities, data, vocab_for_sim="all", tfidf_dir="", target="", min_freq=3):
    alignment = dict()

    if "tfidf" in vocab_for_sim:
        tfidfs = load_tfidf(tfidf_dir, target)
        sorted_tfidfs = sorted(tfidfs.items(), key=lambda x: x[1])

    ## apply the min_freq filter: remove all words that appear less than min_freq times.
    common_words = set()
    for conversation in similarities:
        words = set(similarities[conversation].keys())
        words_to_remove = []
        subset1, subset2 = groupings[conversation]
        for word in similarities[conversation]:
            if len(data[subset1][word]) < min_freq or len(data[subset2][word]) < min_freq:
                words_to_remove.append(word)
        if not common_words:
            common_words = words
        else:
            common_words = common_words.intersection(set(words))
        for word in words_to_remove:
            del similarities[conversation][word]
    if args. verbose:
        print("common words across comparisons for this target:", len(common_words))

    # first check if there are enough targets in all groupings:
    enough_common_lemmas = True
    for conversation in similarities:
        if len(similarities[conversation]) < 3:
            enough_common_lemmas = False
    if not enough_common_lemmas:
        print("skipping one target due to insufficient amount of lemmas")
        return None
    if vocab_for_sim == "all": # pick the same number of words from each conversation across this target. If there are more available, pick the X most frequent ones
        min_num_common_words = min([len(similarities[conversation]) for conversation in similarities])

    for comparison in similarities:
        if "tfidf" in vocab_for_sim:
            N = int(vocab_for_sim.split("_")[1])
            common_lemmas_tfidfs = dict()
            for word in similarities[comparison]:
                wordnopos = word.split("_")[0]
                common_lemmas_tfidfs[word] = tfidfs.get(wordnopos, 0)
            if "reverse" in vocab_for_sim: # the words with LOWEST tfidf
                sorted_by_tfidf = sorted(common_lemmas_tfidfs.items(), key=lambda k: k[1])
            else:
                sorted_by_tfidf = sorted(common_lemmas_tfidfs.items(), key=lambda k: k[1], reverse=True)

            kept = [w for w, v in sorted_by_tfidf[:N]]
            all_sims = [similarities[comparison][word] for word in kept]
            alignment[comparison] = np.average(all_sims)

        elif vocab_for_sim == "all":
            freq_dist = dict()
            for word in similarities[comparison]:
                freq_dist[word] = dict()
                for subset in groupings[comparison]:
                    freq = 0
                    for utt in data[subset][word]:
                        freq += len(utt['lemma_positions'][word])
                    freq_dist[word][subset] = freq

            words = list(freq_dist.keys())
            total_freqs = [sum([freq_dist[word][subset] for subset in freq_dist[word]]) for word in words]

            up_to = min_num_common_words

            sorted_words_freqs = sorted([(t, f) for t, f in zip(words, total_freqs)], key=lambda x: x[1], reverse=True)
            words = [tf[0] for tf in sorted_words_freqs[:up_to]]

            if args.verbose:
                print("considered words:", words, "(original amount of words: ", len(freq_dist.keys()), ")")
            if vocab_for_sim == "all":
                all_sims = [similarities[comparison][word] for word in words]
                all_sims_with_words = {word: similarities[comparison][word] for word in words}
                alignment[comparison] = np.average(all_sims)

    return alignment

def evaluate_stance_alignment_predictions(alignment):
    '''This is for a given target, not a given dataset'''
    # Pairwise accuracy: 4 max
    correct = 0
    for same, diff in [('same-stance_FAVOR', 'different-stance_FAVOR-AGAINST(1)'), ('same-stance_FAVOR', 'different-stance_FAVOR-AGAINST(2)'),
                       ('same-stance_AGAINST', 'different-stance_FAVOR-AGAINST(1)'), ('same-stance_AGAINST', 'different-stance_FAVOR-AGAINST(2)')]:

        if alignment[same] > alignment[diff]:
            correct +=1
    return correct


def significance_difference_test(x, y, alpha=0.05):
    _, normalp = shapiro(np.array(x) - np.array(y))
    if normalp >= alpha:  #  normal so go ahead with t test
        print("Normal! -> ttest")
        _, p = ttest_rel(x, y)
    elif normalp < alpha:  # significant! So we can't do ttest
        print("Not normal! -> Wilcoxon")
        _, p = wilcoxon(x, y)

    d = cohens_d(x, y)
    return p, d



def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff)/ np.std(diff)



def print_highest_lowest_sims(dataset_name, target, equivalences, similarities, tfidf_dir):
    '''This prints the most/least similar words in diff comparisons as a latex table'''
    total_sentences = len(open(tfidf_dir + dataset_name + "/" + target + "/different-stance_FAVOR-AGAINST(1).tsv").readlines()) + len(open(tfidf_dir + dataset_name + "/" + target + "/different-stance_FAVOR-AGAINST(2).tsv").readlines())
    targetname = target.split("%")[0]
    tfidfs = load_tfidf(tfidf_dir + dataset_name + "/", target)
    sorted_tfidfs = sorted(tfidfs.items(), key=lambda x: x[1])

    if dataset_name == "30k":
        targetname = equivalences[targetname]

    for i, pair in enumerate(['different-stance_FAVOR-AGAINST(1)', 'different-stance_FAVOR-AGAINST(2)']):
        sorted_sims = sorted(similarities[pair].items(), key=lambda item: item[1], reverse=True)

        if len(sorted_sims) > 3:
            w1 = sorted_sims[0][0]
            w2 = sorted_sims[1][0]
            w3 = sorted_sims[2][0]
            wm3 = sorted_sims[-3][0].split("_")[0] if sorted_sims[-3][0][0] != "#" else "\\" + \
                                                                                        sorted_sims[-3][0].split("_")[0]
            wm2 = sorted_sims[-2][0].split("_")[0] if sorted_sims[-2][0][0] != "#" else "\\" + \
                                                                                        sorted_sims[-2][0].split("_")[0]
            wm1 = sorted_sims[-1][0].split("_")[0] if sorted_sims[-1][0][0] != "#" else "\\" + \
                                                                                        sorted_sims[-1][0].split("_")[0]

            alltop5words.extend([x[0] for x in sorted_sims[:5]])
            allbottom5words.extend([x[0] for x in sorted_sims[-5:]])


            if i == 0:
                print("\cline{2-5}")
                print("&\multirow{2}{*}{" + targetname + "}  & \multirow{2}{*}{" + str(
                    total_sentences) + "} & " + w1 + ", " + w2 + ", " + w3 + " & " + wm3 + ", " + wm2 + ", " + wm1 + " \\\\")
            else:
                print(" & & & " + w1 + ", " + w2 + ", " + w3 + " & " + wm3 + ", " + wm2 + ", " + wm1 + " \\\\")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="all", help="if 'all', we use all stance datasets. If 'twitter', we use all twitter stance datasets. Alternatively: semeval2016, pstance, covid19, 30k")
    parser.add_argument('--vocab_for_sim', default="all", help="'all', 'tfidf_10' or 'reversetfidf_10'. The number after _ in (reverse)tfidf indicates how many words are used, at most")
    parser.add_argument('--layer', default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--min_freq', default=3, type=int)
    parser.add_argument('--output_similarities', action='store_true', help="For stance data, whether we want to output the words with highest and lowest differences in BETWEEN comparisons (Appendix Table)")  # for the table in the appendix
    args = parser.parse_args()

    tfidf_dir = "dialign_data/"

    if args.output_similarities and args.dataset_name in ["all", "30k"]:
        # We have shorter versions of the target names in the 30k dataset. This is more convenient for the Appendix table in the paper ("We should abolish zoos -> zoos)
        equivalences = {original_shorter.strip().split(",")[0] : original_shorter.strip().split(",")[1] for original_shorter in open("30k_target_names_equivalence.csv").readlines()}
    else:
        equivalences = {}
    if args.dataset_name == "all":
        datasets = ["semeval2016","covid19","pstance","30k"]
    elif args.dataset_name == "twitter":
        datasets = ["semeval2016","covid19","pstance"]
    else:
        datasets = [args.dataset_name]

    ## TODO add comments about what this is
    total_corrects = 0
    accs_by_target = []
    total_pairwise_comparisons = 0
    diffs_same_same = []
    diffs_same_diff = []
    diffs_diff_diff = []

    total_comparisons = 0

    alltop5words = [] # for output_similarities
    allbottom5words = []  # for output_similarities


    for dataset_name in datasets:
        print("Working with dataset:", dataset_name)
        in_dir = "stance_split_data/" + dataset_name + "/"
        middle_dirs = os.listdir(in_dir) # targets
        groupings = {'same-stance_FAVOR': ('FAVOR-1', 'FAVOR-2'), 'same-stance_AGAINST': ('AGAINST-1', 'AGAINST-2'),
                     'different-stance_FAVOR-AGAINST(1)': ('FAVOR-1', 'AGAINST-2'),
                     'different-stance_FAVOR-AGAINST(2)': ('FAVOR-2', 'AGAINST-1')}

        for target in middle_dirs:
            if target:
                if args.verbose:
                    print("Current target:", target)

            # Load vectors and info
            data, vectors = load_vectors_and_info_stance(in_dir + target + "/", args.layer)

            # Calculate similarities
            similarities = calculate_similarities_stance(data, vectors, groupings)

            if args.output_similarities:
                print_highest_lowest_sims(dataset_name, target, equivalences, similarities, tfidf_dir)

            ### Calculate alignment
            alignment = calculate_comparison_similarity(similarities, data, vocab_for_sim=args.vocab_for_sim, tfidf_dir=tfidf_dir + dataset_name + "/", target=target, min_freq=args.min_freq)

            # if it was not possible to calculate it, continue
            if not alignment:
                continue

            total_comparisons += len(similarities)

            # Were within-stance comparisons more aligned than different-stance comparisons?
            correct = evaluate_stance_alignment_predictions(alignment)
            total_corrects += correct
            total_pairwise_comparisons += 4
            accs_by_target.append(correct/4)

            # Collect magnitude of differences on a per-target basis
            diffs_same_same.append(abs(alignment["same-stance_FAVOR"]-alignment["same-stance_AGAINST"]))  # sF - sA
            diffs_diff_diff.append(abs(alignment["different-stance_FAVOR-AGAINST(1)"]-alignment["different-stance_FAVOR-AGAINST(2)"]))
            other_diffs = [abs(alignment["different-stance_FAVOR-AGAINST(1)"]-alignment["same-stance_FAVOR"]), abs(alignment["different-stance_FAVOR-AGAINST(1)"]-alignment["same-stance_AGAINST"]),
                           abs(alignment["different-stance_FAVOR-AGAINST(2)"]-alignment["same-stance_FAVOR"]), abs(alignment["different-stance_FAVOR-AGAINST(2)"]-alignment["same-stance_AGAINST"])]
            diffs_same_diff.append(np.average(other_diffs))


    ######## Evaluation
    print("\n*** RESULTS ***")

    print("-Pairwise accuracy:", total_corrects/total_pairwise_comparisons)

    print("-Differences:")
    print("same vs same:", np.average(diffs_same_same))
    print("diff vs diff:", np.average(diffs_diff_diff))
    print("same vs diff:", np.average(diffs_same_diff))

    # Are differences between comparison types (not sentence sets) significant?
    print("-Statistical significance of differences:")
    p, d = significance_difference_test(diffs_same_same, diffs_same_diff)
    print("p value (same vs same) VS (same vs diff):", p, "effect size (cohen's d):", d)
    p, d = significance_difference_test(diffs_diff_diff, diffs_same_diff)
    print("p value (diff vs diff) VS (same vs diff):", p, "effect size (cohen's d):", d)
    p, d = significance_difference_test(diffs_same_same, diffs_diff_diff)
    print("p value (same vs same) VS (diff vs diff):", p, "effect size (cohen's d):", d)

    print("Total number of comparisons run:", total_comparisons)

    ## Significance of rev-tfidf results: chi-square goodness of fit
    _, p = chisquare(f_obs=[total_corrects, total_pairwise_comparisons-total_corrects], f_exp=[total_pairwise_comparisons/2, total_pairwise_comparisons/2])
    print("-Rev-tfidf chisquare p value:", p)

    # Extra info about the most and least different words
    if args.output_similarities:
        print("proportion of nouns and verbs in the top 5 MOST DIFFERENT words across all targets")
        print("N:", len([w for w in alltop5words if w.endswith("NOUN")])/len(alltop5words), " V:", len([w for w in alltop5words if w.endswith("VERB")])/len(alltop5words))

        print("proportion of nouns and verbs in the top 5 LEAST DIFFERENT words across all targets")
        print("N:", len([w for w in allbottom5words if w.endswith("NOUN")])/len(allbottom5words), " V:", len([w for w in allbottom5words if w.endswith("VERB")])/len(allbottom5words))


