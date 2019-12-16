#!/usr/bin/python3
import numpy as np
import random
import gym

from IPython.display import clear_output
from time import sleep
from io import StringIO
import sys, os

"""
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
Explanation  In this env a taxi must pick up a passenger from one of 4 fixed
pick/drop locations, traverse a 5x5 grid, and drop them off at one of the
other pick/drop locations. The taxi performs one of the following actions at
a time: pickup, dropoff, move left, right, up, or down. The target pickup and
dropoff locations are marked as blue and purple, respectively. Once a
passenger is picked up from the correct location (requiring two time steps)
the taxi color will change to green. An attempt to pickup or drop off the
passenger at the wrong location will incur -10 reward. Also, each move incurs
-1 reward thus encouraging the taxi to minimize time. When the passenger is in the taxi and a pickup occurs, then the env only gives a time penalty


Notes:  This reward structure seems to allow repeated pickup of the passenger
after they are in the taxi. This sitation be an example of how an over
simplified reward signal might allow the system to waste resources, in this case time. 
"""

# Rendering mode; choose from ['human', 'ansi']
MODE = "human"
FPS = 10 
FPS_FACTOR = 30
N_CLIP = 100
N_EPISODES = 3

class State:
    def __init__(self):
        self.taxiX = 3
        self.taxiY = 1
        self.pickup = 2
        self.dropoff = 0    

class PenaltyTypes:
    Time = -1
    WrongDropOrPick = -10

class Actions:
    MoveS = 0
    MoveN = 1
    MoveE = 2
    MoveW = 3
    Pickup = 4
    Dropoff = 5

class TrainingOutputs:
    def __init__(self, frames, epochCount, pickAndDropFailures):
        self.frames = np.array(frames)
        self.epochCount = epochCount
        self.failedPickAndDropCount = np.array(pickAndDropFailures)

def BruteForceSearch(env, maxEpochs=100000):
    r = 0
    penalties = [0,0]
    epochs = 0
    done = False
    frames = []
    while not done and epochs <= maxEpochs:
        # Brute force approach
        #   take random actions until goal is reached
        a = env.action_space.sample()
        s, r, done, info = env.step(a)

        if (r == PenaltyTypes.WrongDropOrPick and a == Actions.Pickup):
            penalties[0] += 1
        if (r == PenaltyTypes.WrongDropOrPick and a == Actions.Dropoff):
            penalties[1] += 1

        frames.append({
            'frame': env.render(mode="ansi"),
            'state': s,
            'action': a,
            'reward': r
            })

        epochs += 1

    return TrainingOutputs(frames, epochs, penalties)

def RenderIPython(outputs):
    print("\ttotal epochs\t\tpenalty rate")

    # temp arrays for stat computations (later)
    e = np.zeros(len(outputs))
    p = np.zeros(len(outputs))

    for i, episode in enumerate(outputs):
        for frame in episode.frames:
            print(frame['frame'])
            print(f"{i}: \t{episode.epochCount}\t\t\t{episode.failedPickAndDropCount.sum() / episode.epochCount:.2f}")
            clear_output(wait=True)
        e[i] = episode.epochCount
        p[i] = episode.failedPickAndDropCount.sum()


    print("------------------------------------------")
    print("\tavg epochs\t\tavg penalties")
    print(f"\t{e.mean():.2f} +/-{e.std():.2f}\t{p.mean():.2f} +/-{p.std():.2f}")

def Render(episodeId, episodeOutput, close=False):
    if close:
        return 

    frames = episodeOutput.frames
    e = episodeOutput.epochCount
    p = episodeOutput.failedPickAndDropCount.sum()

    # just silly - passenger id
    pid = random.randint(500,1000)
    hasPassenger = False

    if MODE != "ansi": # system calls should generally be avoided...
        outfile = sys.stdout
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        outfile = StringIO()

    nclip = min(N_CLIP, len(frames)) 
    for i, frame in enumerate(frames[:nclip]):
        Refresh()
        contents = f"[{episodeId}] frame {i}:\n"
        contents += frame["frame"]
        contents += "\n"
        contents += "\ttotal epochs\t\tpenalty rate\n"
        contents += f"\t{e}\t\t\t{p / e:.2f}\n"

        if (not hasPassenger
            and frame["action"] == Actions.Pickup 
            and frame["reward"] != PenaltyTypes.WrongDropOrPick):
            sleepDur = FPS_FACTOR / FPS
            contents += f"acquired passenger {pid}"
            hasPassenger = True
        else:
            sleepDur = 1 / FPS

        outfile.write(contents)
        outfile.write("\n")

        sleep(sleepDur)

    # render second half of the frames 
    if (nclip == len(frames)):
        return

    Refresh()
    contents = "One moment later...\n"
    outfile.write(contents)
    outfile.write("\n")
    sleep(1)

    mclip = max(nclip, len(frames) - N_CLIP) 
    for i, frame in enumerate(frames[mclip:]):
        Refresh()
        contents = f"[{episodeId}] frame {mclip + i}:\n"
        contents += frame["frame"]
        contents += "\n"
        contents += "\ttotal epochs\t\tpenalty rate\n"
        contents += f"\t{e}\t\t\t{p / e:.2f}\n"

        if (frame["action"] == Actions.Pickup 
            and frame["reward"] != PenaltyTypes.WrongDropOrPick):
            sleepDur = FPS_FACTOR / FPS
            r = frame["reward"]
            contents += f"reward: {r}"
        else:
            sleepDur = 1 / FPS

        outfile.write(contents)
        outfile.write("\n")

        sleep(sleepDur)

    return

def Refresh():
    if MODE != 'ansi':
        os.system('cls' if os.name == 'nt' else 'clear')
    return

def main():
    env = gym.make("Taxi-v3").env
    env.reset()

    s = State()
    # What exactly is this function doing?
    env.s = env.encode(s.taxiX, s.taxiY, s.pickup, s.dropoff)

    print(f"state space: {env.action_space}")
    print(f"obs space: {env.observation_space}")

    outputs = [] 
    for episode in np.arange(N_EPISODES):
        outputs.append(BruteForceSearch(env))

    # Render outputs
    for i, episode in enumerate(outputs):
        Render(i, episode)
        sleep(3)

    # Print some simple stats
    e = np.zeros(len(outputs))
    p = np.zeros(len(outputs))
    for i, episode in enumerate(outputs):
        e[i] = episode.epochCount
        p[i] = episode.failedPickAndDropCount.sum()
    print("------------------------------------------")
    print("\tavg epochs\t\tavg penalties")
    print(f"\t{e.mean():.2f} +/-{e.std():.2f}\t{p.mean():.2f} +/-{p.std():.2f}")

if __name__ == "__main__":
    main()

# Some notes:
# 
# TIL
#   print(f"{var}") is effectively print("{}".format(var))