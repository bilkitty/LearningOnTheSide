def Train(self, envWrapper, agent, verbose=True):
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
            action = agent.GetAction(state, 0)  # TODO: need to "normalize"? hmmm :/
            nextState, reward, done, _ = envWrapper.Step(action)

            agent.SaveExperience(state, action, reward, nextState, done)  # TODO: [expmt] try spacing these out?
            agent.Update(gamma, tau, batchSize)

            epoch += 1
            totalReward += reward
            frames.append({
                'frame': envWrapper.Render(),
                'state': state,
                'action': action,
                'reward': reward})

            if verbose and epoch % (agent.maxEpochs / 1000) == 0:
                RefreshScreen(mode="human")
                s = torch.FloatTensor(state).unsqueeze(0)
                a = torch.FloatTensor(action).unsqueeze(0)
                qv = agent.critic.forward(s, a).detach().squeeze(0).numpy()[0]
                print(f"Training\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

        metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
        episodicMetrics.append(metrics)

    return episodicMetrics, timer() - globalStart


def Test(self, envWrapper, agent, verbose=True):
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
            action = agent.GetAction(state, 0, shouldAddNoise=False)
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
                s = torch.FloatTensor(state).unsqueeze(0)
                a = torch.FloatTensor(action).unsqueeze(0)
                qv = agent.critic.forward(s, a).detach().squeeze(0).numpy()[0]
                print(f"Testing\ne={i}\nr={np.max(reward): 0.2f}\nq={qv: .2f}")

        metrics = Metrics(frames, epoch, timer() - start, totalReward, done)
        episodicMetrics.append(metrics)

    return episodicMetrics, timer() - globalStart