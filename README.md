# Meta-heuristic algorithms to solve the Optimal Reactive Power Dispatch Problem (ORPD)

# Abstract

The Optimal Power Flow (OPF) problem consists on determining the state of an electrical power system that optimizes a given system's performance, satisfying its physical and operational constraints. If the adjusted variables are referent to the system's reactive power, the problem becomes the Optimal Reactive Power Flow (ORPF) or Reactive Power Dispatch problem, which can be mathematically modelled as a non-linear, non-convex, constrained and with discrete and continuous variables optimization problem. Most approaches of the literature do not solve the OPF problem with discrete variables, which are treated as continuous, given the difficulty of solving it considering the discrete nature of those variables. These approaches are unrealistic because real electrical power systems have controls that can only be adjusted by discrete steps. This research work aims to develop and apply to the ORPD problem with discrete variables a solution approach that uses penalty functions to handle the discrete variables, and the problem will be solved by the Grey Wolf Optimizer (GWO). Numerical tests with the IEEE benchmark systems will be performed to validate the developed approach.

# Installation of this repository

1. Install Python through the [Anaconda](https://www.anaconda.com/) distribution.
2. Create a new environment with Python 3.8+:
~~~
conda create -n mhorpd python=3.8 -y
~~~
3. Activate the environment and install the required packages
~~~
conda activate mhorpd
pip install -r requirements.txt
~~~
4. Run the `demo.py` script to visualize the GWO process on the Rosenbrock Function, or the `main.py` script to run the GWO algorithm for the ORPD problem on the IEEE 14-bus system:
~~~
python demo.py
python main.py
~~~

# Publications

Two academic papers were published related to the work on this repository:

"Grey Wolf Optimizer on the solution of the Optimal Reactive Power Dispatch problem with discrete control variables", published on the [LII Brazilian Simposium on Operational Research](https://proceedings.science/sbpo-2020/papers/algoritmo-grey-wolf-optimizer-na-resolucao-do-problema-de-fluxo-de-potencia-otimo-reativo-com-variaveis-de-controle-disc)

"Grey Wolf Optimizer applied to the Rosenbrock Function: a visual demonstration", publiched on the [XXXIII São Paulo State University Math Week proceedings](https://www.fc.unesp.br/Home/Departamentos/Matematica/semanadalicenciatura/caderno-selmat_2020-1.pdf) page 13. 