import numpy as np
import time

def initPopulation(popSize, numVar, numCon, upperBounds, lowerBounds):

    '''
    This function initializes the population array that will be optimized 
    by the chosen meta-heuristic algorithm. 

        ----------------- Arguments -----------------
        * popSize (int32):
            Number of search agents of the algorithm
        
        * numVar (int32):
            Number of variables of the optimization problem
            (dimensionality of the problem)
        
        * numCon (int32):
            Number of constraints of the optimization problem
        
        * upperBounds (numpy.ndarray - float32):
            Numpy array containing the variables upper limits
            $ (numVar, 1) -> must be a rank1 tensor!

        * lowerBounds (numpy.ndarray - float32):
            Numpy array containing the variables lower limits
            $ (numVar, 1) -> must be a rank1 tensor!
    
        ----------------- Returns -----------------
        * popArray (numpy.ndarray - float32):
            Population array with the following dimensionality:
            $ (numVar + numCon + 2, popSize)

            If the optimization problem has no constraints, the
            dimensionality is:
            $ (numVar + 1, popSize)
            In this case there aren't penalization functions or
            a fitness function, therefore, the rows hold the 
            problem variables and the objective function


        ------- Columns -------
        Each column represents a search agent of the algorithm.

        ------- Rows -------
        The rows are divided as follows:

        * Rows 0, 1, ..., numVar - 1: 
            Each of these rows holds one of the variables of the
            optimization problem

        * Row numVar:
            This row holds the objective function of the problem
            calculated using the variables given by the upper
            rows.
        
        * Rows numVar+1 ... numCon-1:
            Each of these rows holds a penalization function for
            the optimization problem constraints (if there are any)
        
        * Row numCon:
            This row holds the fitness function, given by the weighted
            sum of the objetive function and the penalization functions
            for each one of the constraints
    '''

    #Random number generator
    randGen = np.random.default_rng()

    if numCon == 0:
        popArray = np.zeros( shape = (numVar+1, popSize), dtype = np.float32 )

    else:
        popArray = np.zeros( shape = (numVar+numCon+2, popSize), dtype = np.float32 )
    
    # Randomly intializes the variables on the interval [lowerBounds, upperBounds)
    popArray[:numVar, :] = (upperBounds.T - lowerBounds.T)*\
                            randGen.random(size = (numVar, popSize), dtype = np.float32) +\
                            lowerBounds.T

    # Initialize the objective function, penalizations and fitness with a big number
    popArray[numVar:, :] = np.inf

    # Population parameters dictionary
    popParams = {
        'numVar' : numVar,
        'numCon' : numCon,
        'upperBounds' : upperBounds,
        'lowerBounds' : lowerBounds,
        'popSize' : popSize
    }

    return popArray, popParams

def GWO(popArray, popParams, tMax, objectiveFunc, penaltyFunctions = [], penaltyLambdas = []):

    #Random number generator
    randGen = np.random.default_rng()

    # Extract variables from popParams dict
    numVar = popParams['numVar']
    numCon = popParams['numCon']
    upperBounds = popParams['upperBounds'].T
    lowerBounds = popParams['lowerBounds'].T
    popSize = popParams['popSize']

    # Lists to hold the objective function, penalty functions and fitness
    # and a tensor to hold the population array over the iterations
    convObjective = []
    convPenalty = []
    convFitness = []
    popTensor = []
    bestTensor = []

    # Clip the variables to the bounds
    np.clip(a = popArray[:numVar, :], a_min = lowerBounds, 
    a_max = upperBounds, out = popArray[:numVar, :])

    # Initialize the best solutions, i. e., the alpha, beta and delta wolves
    Alpha = np.expand_dims(a = popArray[:numVar, 0].copy(), axis = -1)
    Beta = np.expand_dims(a = popArray[:numVar, 1].copy(), axis = -1)
    Delta = np.expand_dims(a = popArray[:numVar, 2].copy(), axis = -1)

    # Initialize the fitness of the best solutions
    alphaFit = np.inf
    betaFit = np.inf
    deltaFit = np.inf

    # Intialize alpha's penalty and objective functions, just to keep track of them
    alphaObjective = 0.0
    alphaPenalty = 0.0

    # Main loop
    startTime = time.time()

    for t in range(tMax):
        iterationStart = time.time()

        # a(t) -> goes from 2 to 0 over the iterations
        a = 2 - (t*(2/tMax))

        # r1 and r2 -> random numbers between 0 and 1
        r1 = randGen.random(size = (numVar, popSize), dtype = np.float32)
        r2 = randGen.random(size = (numVar, popSize), dtype = np.float32)

        # A(t) -> controls the step size of each wolf in the search space
        A = 2*a*r1 - a

        # C(t) -> controls the movement of each wolf towards the best solutions
        C = 2*r2

        # Calculate the objective function, penalty and fitness
        popArray[numVar, :] = objectiveFunc(popArray[:numVar, :])

        # If there are no penalty functions, the fitness is equal to the objective
        if numCon == 0:
            popArray[-1, :] = popArray[numVar, :].copy()
        else:
            for idx, penaltyFunc in enumerate(penaltyFunctions):
                popArray[numVar + idx + 1, :] = penaltyLambdas[idx]*penaltyFunc(popArray[:numVar, :])
            popArray[-1, :] = np.sum(popArray[numVar:-1, :], axis = 0, keepdims=True)

        # Sort the popArray to get the best solutions of this iteration
        popArray = popArray[:, popArray[-1, :].argsort()]

        # Get the 3 best solutions
        for i in range(3):
            # Update Alpha
            if (popArray[-1, i] <= alphaFit):
                alphaFit = popArray[-1, i].copy()
                Alpha = np.expand_dims(a = popArray[:numVar, i].copy(), axis = -1)

                # Keep track of Alpha's penalty and objective functions
                alphaObjective = popArray[numVar, i].copy()
                alphaPenalty = alphaFit - alphaObjective

            # Update Beta
            if (popArray[-1, i] > alphaFit and popArray[-1, i] <= betaFit):
                betaFit = popArray[-1, i].copy()
                Beta = np.expand_dims(a = popArray[:numVar, i].copy(), axis = -1)

            # Update Delta
            if (popArray[-1, i] > alphaFit and popArray[-1, i] > betaFit and popArray[-1, i] <= deltaFit):
                deltaFit = popArray[-1, i].copy()
                Delta = np.expand_dims(a = popArray[:numVar, i].copy(), axis = -1)
        
        # Keep track of alpha's objective, penalty and fitness functions
        convObjective.append(alphaObjective)
        convPenalty.append(alphaPenalty)
        convFitness.append(alphaFit)

        # Keep track of population and the best solutions over the iterations
        popTensor.append(popArray)
        bestTensor.append(np.hstack((Alpha, Beta, Delta)))

        # Calculate D_alpha, D_beta, D_delta 
        D_alpha = np.abs(C*Alpha - popArray[:numVar, :], dtype = np.float32)
        D_beta = np.abs(C*Beta - popArray[:numVar, :], dtype = np.float32)
        D_delta = np.abs(C*Delta - popArray[:numVar, :], dtype = np.float32)

        # Calculate X_alpha, X_beta, X_delta
        X_alpha = Alpha - A*D_alpha
        X_beta = Beta - A*D_beta
        X_delta = Delta - A*D_delta

        # Update whole population
        popArray[:numVar, :] = (X_alpha + X_beta + X_delta)/3

        # Clip the variables to the bounds
        np.clip(a = popArray[:numVar, :], a_min = lowerBounds, 
        a_max = upperBounds, out = popArray[:numVar, :])

        # Iteration time
        iterationEnd = time.time()
        iterationTime = iterationEnd - iterationStart

        # Show iteration results
        print(f'Iteration {t} -> Alpha fitness = {alphaFit}; Alpha objective = {alphaObjective}; Time = {iterationTime}')

    # Execution time
    endTime = time.time()
    algTime = endTime - startTime

    convCurves = {
        'Obj' : convObjective,
        'Pen' : convPenalty,
        'Fit' : convFitness
    }

    solution = {
        'bestAlpha' : Alpha,
        'bestFit' : alphaFit,
        'bestObj' : alphaObjective,
        'bestPen' : alphaPenalty,
        'time' : algTime
    }

    tensors = {
        'pop' : popTensor,
        'best' : bestTensor
    }

    print('---------------- Solution ----------------')
    print(f'Fitness = {alphaFit}; Objective = {alphaObjective}; Penalty = {alphaPenalty}')
    print(f'Execution time = {algTime}')

    return solution, convCurves, tensors