import numpy as np
from heuristics.gwo import GWO
import rosenbrockProblem as rp

if __name__ == '__main__':
    upB = np.array([
        [1.5, 2.5]
    ])

    lwB = np.array([
        [-1.5, -0.5]
    ])

    gwo_optimizer = GWO(
        population_size = 20,
        dim = 2,
        objective_function = None,
        constraints = None,
        upper_bounds = upB,
        lower_bounds = lwB
    )

    popArray, popParams = mh.initPopulation(20, 2, 0, upB, lwB)
    solution, convCurves, tensors = mh.GWO(popArray, popParams, tMax = 20, objectiveFunc = rp.rosenbrockFunc)

    while (solution['bestObj'] > 1e-4):
        popArray, popParams = mh.initPopulation(20, 2, 0, upB, lwB)
        solution, convCurves, tensors = mh.GWO(popArray, popParams, tMax = 20, objectiveFunc = rp.rosenbrockFunc)

    popTensor = tensors['pop']
    bestTensor = tensors['best']

    x = np.linspace(-1.5, 1.5)
    y = np.linspace(-0.5, 2.5)
    surfaceArray = np.stack((x,y), axis = 0)

    globalOptima = np.array([
        [1.0],
        [1.0],
        [0.0]
    ])

    rp.showAnimation(surfaceArray, globalOptima, popTensor, bestTensor, 
    figSize = (1600, 900), opacity = 0.7, frameDuration = 500)

