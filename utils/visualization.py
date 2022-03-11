import os

import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go


def visualize_fitness(fitness_array, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(0, len(fitness_array), step=1)
    y = fitness_array

    plt.plot(x, y, marker="o", markersize=1, color="#219F94", linewidth=1.5)

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(np.arange(0, np.max(x) + 5, step=5))
    plt.xlabel("Iterations", color="#0F0E0E", fontsize=16)
    plt.ylabel("Fitness", color="#0F0E0E", fontsize=16)
    plt.title(f"Fitness Function", color="#0F0E0E", fontsize=16)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, "fitness.png"), bbox_inches="tight", pad_inches=0
    )
    plt.show()


def visualize_objective(objective_array, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(0, len(objective_array), step=1)
    y = objective_array

    plt.plot(x, y, marker="o", markersize=1, color="#219F94", linewidth=1.5)

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(np.arange(0, np.max(x) + 5, step=5))
    plt.xlabel("Iterations", color="#0F0E0E", fontsize=16)
    plt.ylabel("$f(V, \\theta, t) (MW)$", color="#0F0E0E", fontsize=16)
    plt.title(f"Objective Function", color="#0F0E0E", fontsize=16)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, "objective.png"), bbox_inches="tight", pad_inches=0
    )
    plt.show()


def visualize_constraint(constraint_array, constraint_name, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(0, len(constraint_array), step=1)
    y = constraint_array

    plt.plot(x, y, marker="o", markersize=1, color="#219F94", linewidth=1.5)

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(np.arange(0, np.max(x) + 5, step=5))
    plt.xlabel("Iterations", color="#0F0E0E", fontsize=16)
    plt.ylabel("Constraint", color="#0F0E0E", fontsize=16)
    plt.title(f"Constraint {constraint_name}", color="#0F0E0E", fontsize=16)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, f"constraint_{constraint_name}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()


def visualize_voltages(voltages, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(1, len(voltages) + 1, step=1)
    y = voltages

    plt.scatter(x, y, marker="o", s=6, color="#219F94")

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(x)
    plt.xlabel("Bus", color="#0F0E0E", fontsize=16)
    plt.ylabel("Voltage (pu)", color="#0F0E0E", fontsize=16)
    plt.title(f"Voltage level", color="#0F0E0E", fontsize=16)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, f"voltage.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()


def visualize_voltage_angles(angles, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(1, len(angles) + 1, step=1)
    y = angles

    plt.scatter(x, y, marker="o", s=6, color="#219F94")

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(x)
    plt.yticks(np.arange(np.min(y) - 0.5, np.max(y) + 0.5, step=0.5))
    plt.xlabel("Bus", color="#0F0E0E", fontsize=16)
    plt.ylabel("Voltage angle (°)", color="#0F0E0E", fontsize=16)
    plt.title(f"Voltage angles", color="#0F0E0E", fontsize=16)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, f"voltage_angles.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()


def summary(solutions_dict):
    print(f"Best fitness: {solutions_dict['solution']['fitness']}")
    print(f"Best objective: {solutions_dict['solution']['objective']}")
    print(f"Best constraints: {solutions_dict['solution']['constraints']}")
    print(f"Taps: {solutions_dict['solution']['taps']}")
    print(f"Shunts: {solutions_dict['solution']['shunts']}")
    print(
        f"Time per iteration: {solutions_dict['solution']['time'] / solutions_dict['solution']['iterations']}"
    )
    print(f"Total time: {solutions_dict['solution']['time']}")

    mean_obj = np.mean(solutions_dict["runs"]["objective"])
    max_obj = np.max(solutions_dict["runs"]["objective"])
    std_obj = np.std(solutions_dict["runs"]["objective"])
    print(f"Mean Objective: {mean_obj}")
    print(f"Max Objective: {max_obj}")
    print(f"Std Objective: {std_obj}")
