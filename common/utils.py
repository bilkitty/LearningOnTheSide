import pickle as pkl
import os, sys


def GetScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def GetRootProjectPath():
    return os.environ["PROJECT_ROOT_PATH"]


def GetMaxFloat():
    return float("inf")


def LoadFromPickle(filepath):
    f = open(filepath, "rb")
    contents = pkl.load(f)
    return contents


def SaveAsPickle(contents, filename):
    f = open(filename, "wb")
    pkl.dump(contents, f)
    f.close()


def RefreshScreen(mode):
    if mode != 'ansi':
        os.system('cls' if os.name == 'nt' else 'clear')
