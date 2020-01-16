import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from utils import GetMaxFloat

PLOT_SAMPLE_COUNT = 500
SMOOTHING_WINDOW = 10

# TODO: create frames replay if there's extra time


def PlotPerformanceResults(agentMetrics, actionLabels, name, xMax=GetMaxFloat()):
    """
    inputs:

    return:
        n/a
    """

    if agentMetrics is None:
        print(f"No metrics to plot for '{name}'.")
        return None

    fig = plt.figure(num=1, figsize=(8, 10), dpi=100)
    plt.clf()
    fig.suptitle(name, fontsize=10)
    episodeCount = min(xMax, len(agentMetrics))
    rewards = np.array([x.totalReward for x in agentMetrics][:episodeCount])
    durations = np.array([x.epochCount for x in agentMetrics][:episodeCount])
    actionCounts = np.array([x.actionCounts for x in agentMetrics][:episodeCount])
    assert len(actionCounts[0]) == len(actionLabels), "mismatch between # actions and action labels"
    # Subsample for prettier plots
    plotSampleFreq = episodeCount // min(PLOT_SAMPLE_COUNT, episodeCount)
    actionPlotSampleFreq = episodeCount // min(PLOT_SAMPLE_COUNT // 10, episodeCount)
    rewards = rewards[::plotSampleFreq]
    durations = durations[::plotSampleFreq]
    actionCounts = actionCounts[::actionPlotSampleFreq]
    # Also grab smoothed metrics
    avgRewardsSmoothed = pd.Series(rewards).rolling(SMOOTHING_WINDOW, min_periods=SMOOTHING_WINDOW).mean()
    stdRewardsSmoothed = pd.Series(rewards).rolling(SMOOTHING_WINDOW, min_periods=SMOOTHING_WINDOW).std()

    ax0 = plt.subplot(311)
    ax0.plot(np.linspace(0, episodeCount, episodeCount // plotSampleFreq), avgRewardsSmoothed, linewidth=2)
    ax0.fill_between(np.linspace(0, episodeCount, episodeCount // plotSampleFreq),
                     avgRewardsSmoothed + stdRewardsSmoothed,
                     avgRewardsSmoothed - stdRewardsSmoothed,
                     alpha=0.5)
    ax0.set_xlim(0, episodeCount)
    ax0.set_xlabel("episode")
    ax0.set_ylabel("total reward")
    ax1 = plt.subplot(312)
    ax1.plot(np.linspace(0, episodeCount, episodeCount // plotSampleFreq), durations, linewidth=2)
    ax1.set_xlim(0, episodeCount)
    ax1.set_xlabel("episode")
    ax1.set_ylabel("total epochs")
    ax2 = plt.subplot(313)
    for i, label in enumerate(actionLabels):
        ax2.plot(np.linspace(0, episodeCount, episodeCount // actionPlotSampleFreq),
                 actionCounts[:, i],
                 label=label,
                 linewidth=2)
    ax2.set_xlim(0, episodeCount)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("action count")
    ax2.legend()
    plt.close(fig)
    return fig


def SaveFigure(fig):
    figName = fig._suptitle.get_text().replace(' ', '_')
    print(f"saving '{figName}'")
    fig.savefig(f"{figName}.png")


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))




