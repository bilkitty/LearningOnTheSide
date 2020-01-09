import pickle as pkl

def LoadFromPickle(filepath):
    f = open(filepath, "rb")
    contents = pkl.load(f)
    return contents

def SaveAsPickle(contents, filename):
    f = open(filename, "wb")
    pkl.dump(contents, f)
    f.close()