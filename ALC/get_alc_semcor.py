'''This script is an adapted version of this script: https://github.com/NLPrinceton/ALaCarte/blob/master/alacarte.py '''

from alacarte import *
import pandas as pd


def make_lemma_version(corpus_filename, target, diri):
    infodf = pd.read_csv(diri + target + "/" + target + "_info.csv", sep="\t")
    newdata = []
    with open(corpus_filename) as f:
        for l, (i, r) in zip(f, infodf.iterrows()):
            newline = l.split()[:]
            newline[r['position']] = target.split("_")[0]
            newdata.append(newline)
    with open(diri + target + "/" + target + "_sentences_lemmas.txt", "w") as out:
        for l in newdata:
            out.write(" ".join(l) + "\n")

def main_semcor(args, comm=None):
    '''a la carte embedding induction'''
    rank, size = ranksize(comm)
    matrixfile = args.matrix
    write('Induction Matrix: ' + matrixfile + '\n', comm)
    assert os.path.isfile(matrixfile), "induction matrix must be given if targets given"
    # Load the vectors (only once)
    write('Source Embeddings: ' + args.source + '\n', comm)
    w2v = OrderedDict(load_vectors(args.source))
    M = np.fromfile(matrixfile, dtype=FLOAT)
    d = int(np.sqrt(M.shape[0]))
    assert d == next(iter(w2v.values())).shape[
        0], "induction matrix dimension and word embedding dimension must be the same"
    M = M.reshape(d, d)
    diri = args.corpus_dir + "/"
    print(diri)
    for target in os.listdir(diri):
        print(target)
        if os.path.isdir(diri + target):
            print(target)
            targets = [target.split("_")[0]]
            # Targets have different wordforms - identify them by the lemma
            if diri + target + "/" + target + "_sentences_lemmas.txt" not in os.listdir():
                make_lemma_version(diri + target + "/" + target + "_sentences.txt", target, diri)
            corpusfile = diri + target + "/" + target + "_sentences_lemmas.txt"
            infodf = pd.read_csv(diri + target+"/" + target + "_info.csv", sep="\t")
            indices = infodf["index"]
            for X in [3,5,10,20,25]:
                classes = infodf["X"+str(X)]
                indices1 = [i for i in range(len(classes)) if classes[i] == 1]
                indices2 = [i for i in range(len(classes)) if classes[i] == 2]
                interval1 = [indices1[0], indices1[-1]]
                interval2 = [indices2[0], indices2[-1]]
                alc1 = ALaCarteReader(w2v, targets, wnd=args.window, checkpoint=False, interval=interval1, comm=comm)
                alc2 = ALaCarteReader(w2v, targets, wnd=args.window, checkpoint=False, interval=interval2, comm=comm)
                outtext1 = "part1_X" + str(X)
                outtext2 = "part2_X" + str(X)
                root1 = diri + target + "/" + target + "_" + outtext1 + "_" + args.embeddings_name
                root2 = diri + target + "/" + target + "_" + outtext2 + "_" + args.embeddings_name


                write('Building Context Vectors\n', comm)
                write('Source Corpus: ' + corpusfile + '\n', comm)
                context_vectors1, target_counts1 = corpus_documents(corpusfile, alc1, verbose=args.verbose, comm=comm,
                                                                  english=args.english, lower=args.lower)
                context_vectors2, target_counts2 = corpus_documents(corpusfile, alc2, verbose=args.verbose, comm=comm,
                                                                    english=args.english, lower=args.lower)

                if rank:
                    sys.exit()
                nz1 = target_counts1 > 0
                nz2 = target_counts2 > 0

                Path(root1).parent.mkdir(exist_ok=True)
                Path(root2).parent.mkdir(exist_ok=True)
                write('Dumping Induced Vectors', comm)

                context_vectors1[nz1] = np.true_divide(context_vectors1[nz1], target_counts1[nz1, None], dtype=FLOAT)
                context_vectors2[nz2] = np.true_divide(context_vectors2[nz2], target_counts2[nz2, None], dtype=FLOAT)
                dump_vectors(zip(targets, context_vectors1.dot(M.T)), root1 + '_alacarte.txt')
                dump_vectors(zip(targets, context_vectors2.dot(M.T)), root2 + '_alacarte.txt')


def parse_semcor():
  '''parses command-line arguments'''

  parser = argparse.ArgumentParser(prog='python alacarte.py')

  parser.add_argument('-ename', '--embeddings_name', help='name of the embeddings used, for example 6B.50d', type=str)
  parser.add_argument('--corpus_dir', help='path to the corpus directory (../../semcor_representations)...', type=str)
  parser.add_argument('-m', '--matrix', help='binary file for a la carte transform matrix', type=str)
  parser.add_argument('-v', '--verbose', action='store_true', help='display progess')
  parser.add_argument('-i', '--interval', nargs=2, default=['0', 'inf'], help='corpus position interval')
  parser.add_argument('-s', '--source', default=GLOVEFILE, help='source word embedding file', type=str)
  parser.add_argument('-w', '--window', default=10, help='size of context window', type=int)
  parser.add_argument('-e', '--english', action='store_true', help='check documents for English')
  parser.add_argument('-l', '--lower', action='store_true', help='lower-case documents')
  parser.add_argument('--create-new', action='store_true', help='Also create embeddings for words that are already in'
                                                                'the original embeddings.')
  print(parser)

  return parser.parse_args()



if __name__ == '__main__':

  try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
  except ImportError:
    comm = None


  main_semcor(parse_semcor(), comm=comm)
