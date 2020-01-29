import numpy as np
from timeit import default_timer as timer

from utils import RefreshScreen
from metrics import Metrics

from time import sleep

SCREEN_REFRESH_COUNT = 20
SCREEN_REFRESH_COUNT_EPOCHS = 10000


def Train(envWrapper, agent, verbose=True, headerText=""):
    print(headerText)

    episodicMetrics = []
    globalStart = timer()
    for i in np.arange(agent.maxEpisodes):
        epoch = 0
        frames = []
        done = False
        totalReward = 0
        state = envWrapper.Reset()

        # Console update for progress on # completed rollouts
        if (not verbose) and i % (agent.maxEpisodes / SCREEN_REFRESH_COUNT) == 0:
            print(f"------------ {i/agent.maxEpisodes: .2f}% -----------")

        start = timer()
        while not done and epoch < agent.maxEpochs:
            # Observe change in env
            action = agent.GetAction(state)
            nextState, reward, done, _ = envWrapper.Step(action)
            # TODO: [expmt] try spacing these out?

            # Update agent model
            agent.SaveExperience(state=state, action=action, reward=reward, nextState=nextState, done=done)
            agent.Update()

            # Updates related to env
            epoch += 1
            state = nextState
            totalReward += reward
            frames.append({
                'frame': envWrapper.Render(),
                'state': state,
                'action': action,
                'reward': reward})

            # Console update for individual rollout progress
            if verbose and epoch % (agent.maxEpochs / SCREEN_REFRESH_COUNT_EPOCHS) == 0:
                RefreshScreen(mode="human")
                qv = agent.GetValue(state, action)
                print(headerText)
                print(f"Training\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

        metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
        episodicMetrics.append(metrics)

    return episodicMetrics, timer() - globalStart


def Test(envWrapper, agent, verbose=True, headerText=""):
    print(headerText)

    episodicMetrics = []
    globalStart = timer()
    for i in np.arange(agent.maxEpisodes):
        epoch = 0
        totalReward = 0
        frames = []
        done = False
        state = envWrapper.Reset()

        # Console update for progress on # completed rollouts
        if (not verbose) and i % (agent.maxEpisodes / SCREEN_REFRESH_COUNT) == 0:
            print(f"------------ {i/agent.maxEpisodes: .2f}% -----------")

        start = timer()
        while not done and epoch < agent.maxEpochs:
            # Observe env and conduct policy
            action = agent.GetBestAction(state)
            nextState, reward, done, _ = envWrapper.Step(action)

            epoch += 1
            state = nextState
            totalReward += reward
            frames.append({
                'frame': envWrapper.Render(),
                'state': state,
                'action': action,
                'reward': reward})

            if verbose and epoch % (agent.maxEpochs / SCREEN_REFRESH_COUNT_EPOCHS) == 0:
                RefreshScreen(mode="human")
                qv = agent.GetValue(state, action)
                print(headerText)
                print(f"Testing\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

        metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
        episodicMetrics.append(metrics)

    return episodicMetrics, timer() - globalStart
