import numpy as np
from matplotlib import pyplot as plt

PLOT_SAMPLE_FREQ = 5000


def PlotPerformanceResults(agentMetrics):
    """
    inputs:

    return:
        n/a
    """

    fig = plt.figure(num=1, figsize=(8,10), dpi=100)
    rewards = [x.totalReward for x in agentMetrics]
    durations = [x.epochCount for x in agentMetrics]
    actionCounts = [x.actionCount for x in agentMetrics]
    plt.plot(a0, color='red', marker='o', markersize=2, linewidth=0, label="s", markevery=PLOT_SAMPLE_FREQ//10)
    plt.plot(a1, color='green', marker='o', markersize=2, linewidth=0, label="n", markevery=PLOT_SAMPLE_FREQ//10 + 5)
    plt.plot(a2, color='blue', marker='o', markersize=2, linewidth=0, label="e", markevery=PLOT_SAMPLE_FREQ//10 + 15)
    plt.plot(a3, color='orange', marker='o', markersize=2, linewidth=0, label="w", markevery=PLOT_SAMPLE_FREQ//10 + 25)
    plt.plot(a4, color='magenta', marker='x', markersize=2, linewidth=0, label="pick", markevery=PLOT_SAMPLE_FREQ//10 + 35)
    plt.plot(a5, color='cyan', marker='d', markersize=2, linewidth=0, label="drop", markevery=PLOT_SAMPLE_FREQ//10 + 45)
    plt.xlabel("episode")
    plt.ylabel("action count")
    plt.legend()

    plt.subplot(221)
    plt.plot(rewards, linewidth=0, marker="o", markevery=PLOT_SAMPLE_FREQ)
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.subplot(222)
    plt.plot(durations, linewidth=0, marker="o", markevery=PLOT_SAMPLE_FREQ)
    plt.xlabel("episode")
    plt.ylabel("total epochs")
    plt.plot(actionCounts, cmap="warm", linewidth=2, label="drop", markevery=PLOT_SAMPLE_FREQ//10 + 45)
    plt.xlabel("episode")
    plt.ylabel("action count")
    plt.legend()
    return fig


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




