import pickle as pkl
import json
import os, sys

"""
Commandline Helpers
"""

def LoadFromJsonFile(jsonfilepath):
    with open(jsonfilepath, "rb") as f:
        _s = f.read()
        argmap = json.loads(_s.decode())
        assert isinstance(argmap, dict), "expected dict to load"

    return argmap

def SaveAsJsonFile(argmap, jsonfilepath):
    assert isinstance(argmap, dict), "expected dict to save"
    with open(jsonfilepath, "wb") as f:
        _s = json.dump(argmap)
        f.write(str.encode(_s))

"""
Path Helpers
"""

def GetScriptPath():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def GetRootProjectPath():
    assert "PROJECT_ROOT_PATH" in os.environ.keys()
    return os.environ["PROJECT_ROOT_PATH"]

"""
Checkpoint Helpers
"""

def LoadFromPickle(filepath):
    f = open(filepath, "rb")
    contents = pkl.load(f)
    return contents


def SaveAsPickle(contents, filepath):
    f = open(filepath, "wb")
    pkl.dump(contents, f)
    f.close()

"""
Rendering Helpers
"""

def RefreshScreen(mode):
    if mode != 'ansi':
        os.system('cls' if os.name == 'nt' else 'clear')

"""
Misc 
"""

def GetMaxFloat():
    return float("inf")
