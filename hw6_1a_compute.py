from platypus.algorithms import *
from platypus.problems import DTLZ2
from platypus.indicators import Hypervolume
from platypus.experimenter import experiment, calculate, display

if __name__ == "__main__":
    #replace MOEAD by NSGAII
    algorithms = [MOEAD, (NSGAIII, {"divisions_outer":12}),GDE3]
    problems = [DTLZ2(2)]

    # run the experiment
    results = experiment(algorithms, problems, seeds=30, nfe=5000)

    # calculate the hypervolume indicator
    hyp = Hypervolume(minimum=[0, 0, 0], maximum=[1, 1, 1])
    hyp_result = calculate(results, hyp)
    display(hyp_result, ndigits=3)
    