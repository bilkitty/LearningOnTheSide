import numpy as np
from timeit import default_timer as timer

from utils import RefreshScreen
from metrics import Metrics


def Train(envWrapper, agent, verbose=True):

    episodicMetrics = []
    globalStart = timer()
    for i in np.arange(agent.maxEpisodes):
        epoch = 0
        totalReward = 0
        frames = []
        done = False
        state = envWrapper.Reset()
        start = timer()
        while not done and epoch < agent.maxEpochs:
            action = agent.GetAction(state)
            nextState, reward, done, _ = envWrapper.Step(action)
            # TODO: [expmt] try spacing these out?
            agent.SaveExperience(state=state, action=action, reward=reward, nextState=nextState, done=done)
            agent.Update()

            epoch += 1
            totalReward += reward
            frames.append({
                'frame': envWrapper.Render(),
                'state': state,
                'action': action,
                'reward': reward})

            if verbose and epoch % (agent.maxEpochs / 1000) == 0:
                RefreshScreen(mode="human")
                qv = agent.GetValue(state, action)
                print(f"Training\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

        metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
        episodicMetrics.append(metrics)

    return episodicMetrics, timer() - globalStart


def Test(envWrapper, agent, verbose=True):

    episodicMetrics = []
    globalStart = timer()
    for i in np.arange(agent.maxEpisodes):
        epoch = 0
        totalReward = 0
        frames = []
        done = False
        state = envWrapper.Reset()
        start = timer()
        while not done and epoch < agent.maxEpochs:
            action = agent.GetBestAction(state)
            nextState, reward, done, _ = envWrapper.Step(action)

            epoch += 1
            totalReward += reward
            frames.append({
                'frame': envWrapper.Render(),
                'state': state,
                'action': action,
                'reward': reward})

            if verbose and epoch % (agent.maxEpochs / 1000) == 0:
                RefreshScreen(mode="human")
                qv = agent.GetValue(state, action)
                print(f"Testing\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

        metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
        episodicMetrics.append(metrics)

    return episodicMetrics, timer() - globalStart
