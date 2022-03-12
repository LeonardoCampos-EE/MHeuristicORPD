import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc


rc("text", usetex=True)
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})


def visualize_fitness(fitness_array, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(0, len(fitness_array), step=1)
    y = np.asarray(fitness_array) * 100

    plt.plot(x, y, marker="o", markersize=1, color="#219F94", linewidth=1.5)

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(np.arange(0, np.max(x) + 5, step=5))
    plt.xlabel("Iterações", color="#0F0E0E", fontsize=14)
    plt.ylabel(r"$FA(V, \theta, t, b^{sh})$", color="#0F0E0E", fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(os.path.join(output_dir, "fitness.png"))
    # plt.show()


def visualize_objective(objective_array, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(0, len(objective_array), step=1)
    y = np.asarray(objective_array) * 100

    plt.plot(x, y, marker="o", markersize=1, color="#219F94", linewidth=1.5)

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(np.arange(0, np.max(x) + 5, step=5))
    plt.xlabel("Iterações", color="#0F0E0E", fontsize=14)
    plt.ylabel(r"$f(V, \theta, t) (MW)$", color="#0F0E0E", fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(os.path.join(output_dir, "objective.png"))
    # plt.show()


def visualize_constraint(
    constraint_array, constraint_name, constraint_text, output_dir
):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(0, len(constraint_array), step=1)
    y = constraint_array

    plt.plot(x, y, marker="o", markersize=1, color="#219F94", linewidth=1.5)

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(np.arange(0, np.max(x) + 5, step=5))
    plt.xlabel("Iterações", color="#0F0E0E", fontsize=14)
    plt.ylabel(constraint_text, color="#0F0E0E", fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, f"constraint_{constraint_name}.png"),
    )
    # plt.show()


def visualize_voltages(voltages, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(1, len(voltages) + 1, step=1)
    y = voltages

    plt.scatter(x, y, marker="o", s=6, color="#219F94")

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(x)
    plt.xlabel("Barra", color="#0F0E0E", fontsize=14)
    plt.ylabel(r"V (pu)", color="#0F0E0E", fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, f"voltage.png"),
    )
    # plt.show()


def visualize_voltage_angles(angles, output_dir):

    fig, ax = plt.subplots(1, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(1, len(angles) + 1, step=1)
    y = angles

    plt.scatter(x, y, marker="o", s=6, color="#219F94")

    ax.tick_params(axis="both", colors="#0F0E0E")
    plt.xticks(x)
    plt.yticks(np.arange(np.min(y) - 0.5, np.max(y) + 0.5, step=0.5))
    plt.xlabel("Barra", color="#0F0E0E", fontsize=14)
    plt.ylabel(r"$\theta (^\circ)$", color="#0F0E0E", fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#0F0E0E")
    ax.spines["bottom"].set_color("#0F0E0E")

    ax.set_axisbelow(True)
    plt.grid(True)
    fig.savefig(
        os.path.join(output_dir, f"voltage_angles.png"),
    )
    # plt.show()


def summary(solutions_dict):
    print(f"Best fitness: {solutions_dict['solution']['fitness']*100}")
    print(f"Best objective: {solutions_dict['solution']['objective']*100}")
    print(f"Best constraints: {solutions_dict['solution']['constraints']}")
    print(f"Taps: {solutions_dict['solution']['taps']}")
    print(f"Shunts: {solutions_dict['solution']['shunts']}")
    print(
        f"Time per iteration: {solutions_dict['solution']['time'] / solutions_dict['solution']['iterations']}"
    )
    print(f"Total time: {solutions_dict['solution']['time']}")

    mean_obj = np.mean(np.array(solutions_dict["runs"]["objective"])*100)
    max_obj = np.max(np.array(solutions_dict["runs"]["objective"])*100)
    std_obj = np.std(np.array(solutions_dict["runs"]["objective"])*100)
    print(f"Max Objective: {max_obj}")
    print(f"Mean Objective: {mean_obj}")
    print(f"Std Objective: {np.format_float_scientific(std_obj)}")
