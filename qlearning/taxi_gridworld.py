#!/usr/bin/python3
import numpy as np
import pickle as pkl
import random
import math
import gym

from IPython.display import clear_output
from time import sleep
from timeit import default_timer as timer
from io import StringIO
import sys, os

from matplotlib import pyplot as plt

# Rendering mode; choose from ['human', 'ansi']
MODE = "human"
FPS = 10 
FPS_FACTOR = 30
N_CLIP = 20
MAX_TRAINING_EPISODES = 10
MAX_TRAINING_EPOCHS = 10000

# Qlearning params
LR_ALPHA = 0.1
DR_GAMMA = 0.6
EPSILON = 0.1
QTABLE_FILE = "qtable.pkl"
POLICY_FILE = "policy.pkl"
SHOULD_RECYCLE = False 
SHOULD_PLOT = True

#
# Tutorial
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
#

"""
Summary of required packages
python3 -m pip install gym, numpy, decorator, matplotlib
sudo apt-get install python3.6-tk

Summary of key functions


BruteForceSearch(env, maxEpisodes=100, maxEpochs=100000)
    Performs random actions until goal state is reached.

Parameters
    env             openai gym environment object
    maxEpisodes     max number of episodes
    maxEpochs       max number of epochs
Returns
    Eval outputs    frames, epoch and penalty counts, etc.

LearnPolicy(env, maxEpisodes=100, maxEpochs=10000)
    Estimates q-values for action state pairs via q-learning algorithm.

Parameters
    env             openai gym environment object
    maxEpisodes     max number of episodes
    maxEpochs       max number of epochs
Returns
    2d array        a q-table of size m states by n actions
    Eval outputs    frames, epoch and penalty counts, etc.

ExecutePolicy(env, qTable, maxEpisodes=100, maxEpochs=100000)
    Uses learnt q-values to exploit best actions to find goal state. 

Parameters
    env             openai gym environment object
    qTable          a 2d table of q-values for state-action pairs
    maxEpisodes     max number of episodes
    maxEpochs       max number of epochs
Returns
    Eval outputs    frames, epoch and penalty counts, etc.

"""


"""
Learning

Bruteforce Search
Actions are taken randomly until the task is complete or timeout. This mehtod 
completely ignores the env's reward.

Q-learning 
The reward that results from a particular action-state pair is
logged in the form of a q-value table. Each value in the table corresponds to
the "reprentative quality" of an action taken from that state. In other words,
the q-value captures a notion of "goodness/badness" of an action specific to
an instance of the environment.

Once learnt through enough random exploration of actions, the Q-values tend to
converge. If they do, then they will provide an action-value function which
the agent can exploit to pick the most optimal action from a given state. To
prevent overfitting, the system must balance a tradeoff between exploration
and exploitation. The parameter epsilon controls this tradeoff. Lower epsilon 
values favour exploration.

The general update formula: 
Q(a,s) is initialized arbitrarily 
Q(a,s)  <- (1-A) * Q(a,s) + A * [reward + G * max_a(Q(all a, next s))],
where A is the learning rate and G is the discount rate, both in [0,1]

In words, the action value, Q, is a weighted average of the prev action
value and the maximal discounted future reward. The window for future
reward is technically infinite, but parameter G controls the effective
relevance of future rewards. If small, then the agent will learn
myopically. Otherwise, it will take actions that consider longer-term
effects.

In this example, the data structure we'll use for q-value storage is a table.
The table's dimensions are M=size of STATE SPACE by N=size of ACTION SPACE

Interesting experiments:
    selection of initial values - in the example, zero initialization is used. WHy not stochastic? What if we initalize with uniformly random values? Gaussian? Another distro?

"""

class EvaluationOutputs:
    def __init__(self, frames, epochs, penalties, totalReward, success):
        self.frames = frames
        self.epochCount = epochs
        self.failedPickAndDropCount = np.array(penalties)
        self.totalReward = totalReward
        self.success = success
        self.actions = [] 

    def SetActions(self, actions):
        self.actions = actions

    def Str(self):
        s = ""
        for frame in self.frames[len(self.frames) - 5:]:
            s += frame["frame"]
            s += "\n"
        return f"{s}\ne: {self.epochCount}\np: {self.failedPickAndDropCount}\nr: {self.totalReward}\nsuccess: {self.success}\n"

def BruteForceSearch(env, maxEpisodes=100, maxEpochs=100000):
    evaluationResults = []
    for episode in np.arange(maxEpisodes):
        r = 0
        totalReward = 0
        penalties = [0,0]
        epochs = 0
        done = False
        frames = []
        while not done and epochs <= maxEpochs:
            a = env.action_space.sample()
            s, r, done, info = env.step(a)

            if (r == PenaltyTypes.WrongDropOrPick and a == Action.Pickup):
                penalties[0] += 1
            if (r == PenaltyTypes.WrongDropOrPick and a == Action.Dropoff):
                penalties[1] += 1

            frames.append({
                'frame': env.render(mode="ansi"),
                'state': s,
                'action': a,
                'reward': r
                })

            epochs += 1
            totalReward += r

        evaluationResults.append(EvaluationOutputs(frames, epochs, penalties, totalReward, done))

    return evaluationResults

def CreateQTable(m,n):
    # TODO: explore different initialization schemes - e.g., uniformly random, gaussian, other, etc.
    return np.zeros([m,n])

def LoadQTable(filepath):
    f = open(filepath, "rb")
    qtable = pkl.load(f)
    # some checks
    return qtable

# TODO:
# plotting idea -  a heatmap of states represented as 5x5 grid?
#                  create 6 plots for each action where each plot is a 5x5 2d heatmap showing the qvalue for a particular action
#                  not sure how to reshape the rows (states) into the grid
#                      25 possible grid positions
#                      4 possible correct pickup locations     -> 100 states; passenger is waiting for pickup
#                      4 possible target drop off locations    -> 400 states; passenger is in vehicle 
#                      
def LearnPolicy(env, maxEpisodes=100, maxEpochs=10000):
    qTable = LoadQTable(QTABLE_FILE) if SHOULD_RECYCLE else CreateQTable(env.observation_space.n, env.action_space.n)
    textinfo = f"qt: {qTable.shape}\nProgress...\n"
    timings = ""
    
    trainingResults = []
    for i in range(0, maxEpisodes):
        r = 0
        totalReward = 0
        epochs = 0
        penalties = [0,0]
        frames = []
        done = False
        s = env.reset() 
        
        # extras
        start = timer()
        ahist = np.zeros(env.action_space.n)
        while not done and epochs < maxEpochs:
            if random.uniform(0, 1) < EPSILON:
                a = env.action_space.sample()
            else:
                a = np.argmax(qTable[s])        # Get maximizing parameter 

            # Update qtable after taking action
            sNext, r, done, info = env.step(a)

            q = qTable[s,a]
            qMaxFuture = np.max(qTable[sNext]) # Get maximal value

            qTable[sNext,a] = (1 - LR_ALPHA) * q + LR_ALPHA * (r + DR_GAMMA * qMaxFuture)

            s = sNext

            # log number epochs, penalties, and action counts 
            if (r == PenaltyTypes.WrongDropOrPick and a == Action.Pickup):
                penalties[0] += 1
            if (r == PenaltyTypes.WrongDropOrPick and a == Action.Dropoff):
                penalties[1] += 1

            ahist[a] += 1
            if (epochs % (maxEpochs / 2) == 0):
                Refresh()
                print(f"Training\ne={i}\nr={r}\nq={qTable[s,a]: .2f}")
                totalCount = ahist.sum()
                for b, cnt in enumerate(ahist): print(f"a{b}  {cnt/totalCount: .4f}")
                print(f"\n{timings}")

            epochs += 1

        timings = textinfo + f"{i}: elapsed {timer() - start: 0.2f}s\n"
        trainingResults.append(EvaluationOutputs(frames, epochs, penalties, totalReward, done))

    return qTable, trainingResults

def ExecutePolicy(env, qTable, maxEpisodes=100, maxEpochs=100000):
    evaluationResults = []
    for i in range(maxEpisodes):
        r = 0
        epochs = 0
        penalties = [0,0]
        frames = []
        done = False
        s = env.reset() 
        
        start = timer()
        ahist = np.zeros(env.action_space.n)
        totalReward = 0
        while not done and epochs < maxEpochs:
            # Exploit only
            a = np.argmax(qTable[s])
            s, r, done, info = env.step(a)

            # Update outputs
            if (r == PenaltyTypes.WrongDropOrPick and a == Action.Pickup):
                penalties[0] += 1
            if (r == PenaltyTypes.WrongDropOrPick and a == Action.Dropoff):
                penalties[1] += 1

            frames.append({
                'frame': env.render(mode="ansi"),
                'state': s,
                'action': a,
                'reward': r
                })


            ahist[a] += 1
            if (epochs % (maxEpochs * 2) == 0):
                Refresh()
                print(f"Evaluating\ne={i}\nr={r}\nq={qTable[s,a]: .2f}")
                totalCount = ahist.sum()
                for b, cnt in enumerate(ahist): print(f"a{b}  {cnt/totalCount: .4f}")

            epochs += 1
            totalReward += r
        res = EvaluationOutputs(frames, epochs, penalties, totalReward, done)
        res.SetActions(ahist)
        evaluationResults.append(res)

    return evaluationResults

"""
Environment

Explanation  In this env a taxi must pick up a passenger from one of 4 fixed
pick/drop locations, traverse a 5x5 grid, and drop them off at one of the
other pick/drop locations. The taxi performs one of the following actions at
a time: pickup, dropoff, move left, right, up, or down. The target pickup and
dropoff locations are marked as blue and purple, respectively. Once a
passenger is picked up from the correct location (requiring two time steps)
the taxi color will change to green. An attempt to pickup or drop off the
passenger at the wrong location will incur -10 reward. Also, each move incurs
-1 reward thus encouraging the taxi to minimize time. When the passenger is in
the taxi and a pickup (incorrectly) occurs, then the env only gives a time penalty.
The agent receives reward of 20 for successfully dropping off the passenger.

Action space
    There are six total possible actions. Obviously, only one action is taken at a time: 
    4 directions of movement + pickup + dropoff.
    

State space
    There are 500 total possible states, including when the passenger is in the car: 
    5x5 grid positions x (4 + 1) passenger positions x 4 destinations 
    Really, each row is a summarization of the car's and passenger's possible position in the grid. It 
    doesn't clearly map to a particular visual representation of the grid. Or maybe we just need to find the
    appropriate mapping.

Notes:  This reward structure seems to allow repeated pickup of the passenger
after they are in the taxi. This sitation be an example of how an over
simplified reward signal might allow the system to waste resources, in this case time. 
"""

#
# TODO: make taxi grid world class so that main.py can invoke it. 
#
#class TaxiGridWorld():

class State:
    posR = 0
    posG = 1
    posY = 2
    posB = 3
    onBoard = 4
    def __init__(self):
        self.taxiY = State.posB 
        self.taxiX = State.posG  
        self.pickup = State.posY  
        self.dropoff = State.posR 
    
    @staticmethod
    # Avoiding dependence on array indexing to hopefully mitigate bugs in future
    def Enumerate():
        return [(State.posR, "R"), (State.posG, "G"), (State.posY, "Y"), (State.posB, "B"), (State.onBoard, "OnBoard")]

class PenaltyTypes:
    Time = -1
    WrongDropOrPick = -10

class Action:
    MoveS = 0
    MoveN = 1
    MoveE = 2
    MoveW = 3
    Pickup = 4
    Dropoff = 5

    @staticmethod
    # Avoiding dependence on array indexing to hopefully mitigate bugs in future
    def Enumerate():
        return [(Action.MoveS, "v"), (Action.MoveN, "^"), 
                (Action.MoveE, ">"), (Action.MoveW, "<"), 
                (Action.Pickup, "P"), (Action.Dropoff, "D")]

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

def Render(extantDisplay, episodeOutput, close=False):
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
        contents = f"{extantDisplay} frame {i}:\n"
        contents += frame["frame"]
        contents += "\n"
        contents += "\ttotal epochs\t\tpenalty rate\n"
        contents += f"\t{e}\t\t\t{p / e:.2f}\n"

        if (not hasPassenger
            and frame["action"] == Action.Pickup 
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
        contents = f"{extantDisplay} frame {mclip + i}:\n"
        contents += frame["frame"]
        contents += "\n"
        contents += "\ttotal epochs\t\tpenalty rate\n"
        contents += f"\t{e}\t\t\t{p / e:.2f}\n"

        if (frame["action"] == Action.Pickup 
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

# TODO: plot all rewards and avg
#       plot histogram of action counts
#       plot the length (epoch count) per episode
def SaveAsPickle(contents, filename):
    f = open(filename, "wb")
    pkl.dump(contents, f)
    f.close()

def main():
    env = gym.make("Taxi-v3").env
    env.reset()

    s = State()
    # What exactly is this function doing?
    env.s = env.encode(s.taxiY, s.taxiX, s.pickup, s.dropoff)

    infoText = f"state space: {env.action_space}\n"
    infoText += f"obs space: {env.observation_space}\n"

    qtable = None
    if (len(sys.argv) > 1 and sys.argv[1] == "0"):
        print("Bruteforcing it")
        allOutputs = BruteForceSearch(env)
    elif (len(sys.argv) > 1 and sys.argv[1] == "1"):
        print("Q-learning it")
        if (os.path.exists(POLICY_FILE)):
            qtable = LoadQTable(POLICY_FILE)
            print("Loaded policy")
        else:
            qtable, trainingOutput = LearnPolicy(env, MAX_TRAINING_EPISODES, MAX_TRAINING_EPOCHS)
            SaveAsPickle(qtable, POLICY_FILE)
            SaveAsPickle(trainingOutput, "train.pkl")
            print("Finished policy training")

        assert qtable is not None, "Failed to create qtable"
        allOutputs = ExecutePolicy(env, qtable, MAX_TRAINING_EPISODES, MAX_TRAINING_EPOCHS)
        SaveAsPickle(allOutputs, "eval.pkl")
        print("Finished execution")
    else:
        print("udk wtf i want")
        return

    if not SHOULD_PLOT:
        return

    plt.figure(num=1, figsize=(8,8), dpi=100)
    a0 = [x.actions[Action.MoveS] for x in allOutputs]
    a1 = [x.actions[Action.MoveN] for x in allOutputs]
    a2 = [x.actions[Action.MoveE] for x in allOutputs]
    a3 = [x.actions[Action.MoveW] for x in allOutputs]
    a4 = [x.actions[Action.Pickup] for x in allOutputs]
    a5 = [x.actions[Action.Dropoff] for x in allOutputs]
    plt.plot(a0, color='red', linewidth=1, label="s")
    plt.plot(a1, color='orange', linewidth=1, label="n")
    plt.plot(a2, color='green', linewidth=1, label="e")
    plt.plot(a3, color='blue', linewidth=1, label="w")
    plt.plot(a4, color='magenta', marker='x', linewidth=0, label="pick")
    plt.plot(a5, color='magenta', marker='o', linewidth=0, label="drop")
    plt.xlabel("episode")
    plt.ylabel("action count")
    plt.legend()
    plt.savefig("episode_action_counts.png")

    plt.figure(num=1, figsize=(8,8), dpi=100)
    rewards = [x.totalReward for x in allOutputs]
    durations = [x.epochCount for x in allOutputs]
    penalties = [x.failedPickAndDropCount.sum() for x in allOutputs]
    plt.subplot(221)
    plt.plot(rewards)
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.subplot(222)
    plt.plot(durations)
    plt.xlabel("episode")
    plt.ylabel("total epochs")

    stateNames = [x[1] for x in State.Enumerate()]
    actionNames = [x[1] for x in Action.Enumerate()]
    for s in range(qtable.shape[0]):
        bestActGrids = []
        bestActions = np.array([np.argmax(s) for s in qtable])
        for x in range(20):
            bestActGrids.append(bestActions[x::20])

        assert len(bestActGrids) == 20, f"Unexpected no. best action grids {len(bestActGrids)}"
        assert bestActGrids[0].shape[0] == 25, f"Unexpected dimension of best action grid {bestActGrids[0].shape[0]}"

    # Show best action per grid position
    # Show qvalue per grid position for each action
    sp2 = plt.subplot(223)
    sp3 = plt.subplot(224)
    for a in range(qtable.shape[1]):
        qValGrids = []
        stateVals = qtable[:,a]
        for x in range(20):
            qValGrids.append(stateVals[x::20])
            
        assert len(qValGrids) == 20, f"Unexpected no. qValGrids {len(qValGrids)}"
        assert qValGrids[0].shape[0] == 25, f"Unexpected dimension of action-state grid {qValGrids[0].shape[0]}"
        for i, qvg in enumerate(qValGrids):
            passengerLoc = stateNames[i // 4]
            destinationLoc = stateNames[i % 4]

            bag = np.array(bestActGrids[i])
            sp2.imshow(np.ones((5, 5)), cmap="Greys")
            sp2.text(0,4, f"Y\n")       # bottom left corner
            sp2.text(0,0, f"R\n")       # top left corner
            sp2.text(4,0, f"G\n")       # top right corner
            sp2.text(3,4, f"B\n")       # 1 unit left of the bottom right corner
            for j in range(len(bag)):
                xj = j % 5
                yj = j // 5
                jj = 5 * yj + xj
                bbox = dict(facecolor='green', alpha=0.5) if bag[jj] == a else None
                sp2.text(xj, yj, f"{actionNames[bag[jj]]}: {qtable[jj][bag[jj]]:.2f}", fontsize=6, bbox=bbox)
            sp2.set(xlabel=f"'{passengerLoc}' -> '{destinationLoc}'")
            
            qvg = np.array(qvg)
            sp3.imshow(qvg.reshape(5, 5), cmap="hot")
            sp3.text(0,4, f"Y\n")       # bottom left corner
            sp3.text(0,0, f"R\n")       # top left corner
            sp3.text(4,0, f"G\n")       # top right corner
            sp3.text(3,4, f"B\n")       # 1 unit left of the bottom right corner
            for j in range(len(qvg)):
                xj = j % 5
                yj = j // 5
                jj = 5 * yj + xj
                sp3.text(xj, yj, f"{round(qvg[jj],2)}", fontsize=6)
            sp3.set(xlabel=f"'{passengerLoc}' -> '{destinationLoc}', max qval={qvg.max():.3f}")
            
            plt.savefig(f"{actionNames[a]}_pass{passengerLoc}_dest{destinationLoc}.png")
            sp2.cla()
            sp3.cla()
           

if __name__ == "__main__":
    main()

# Some notes:
# 
# TIL
#   print(f"{var}") is effectively print("{}".format(var)) requires 3.6+