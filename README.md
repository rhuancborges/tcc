# RKO - Random-Key Optimizer (Python Framework)
1. Introduction to the Random-Key Optimizer (RKO)
The Random-Key Optimizer (RKO) is a versatile and efficient metaheuristic framework designed for a wide range of combinatorial optimization problems. Its core paradigm is the encoding of solutions as vectors of random keys—real numbers uniformly distributed in the interval [0, 1). This representation maps the discrete, and often complex, search space of a combinatorial problem to a continuous n-dimensional unit hypercube.

The primary strength of the RKO framework lies in its modular architecture, which decouples the search algorithms from the problem-specific logic. This is achieved through a problem-specific decoder, a user-defined function that translates a random-key vector into a feasible solution for the target problem.

This design allows for the seamless integration of multiple classic metaheuristics (e.g., Simulated Annealing, Iterated Local Search, Genetic Algorithms) that can operate independently or in parallel. When executed concurrently, these algorithms share high-quality solutions through a common elite solution pool, fostering a collaborative and robust search process.

2. Framework Architecture
The Python implementation of RKO is centered around the RKO class, which encapsulates all search operators and metaheuristics.

Core Operators: The framework includes a set of powerful, problem-independent operators that work on the random-key vectors:

shaking: A diversification mechanism to escape local optima by applying controlled perturbations to a solution.

RVND (Random Variable Neighborhood Descent): An intensification strategy that systematically explores multiple neighborhood structures (SwapLS, FareyLS, InvertLS, NelderMeadSearch) in a randomized order to find local optima.

Blending: A crossover operator for population-based methods that combines two parent solutions to generate offspring.

Parallel Execution: The solve method orchestrates the parallel execution of multiple metaheuristic workers using Python's multiprocessing library. These workers operate on the same problem instance and share their findings through a thread-safe SolutionPool.

3. How to Use the RKO Framework
The main workflow consists of three steps:

Define the Problem Environment: Create a Python class that contains all the logic specific to your optimization problem.

Instantiate the RKO Solver: Create an instance of the RKO class, providing your custom environment object and a total time limit.

Execute the Solver: Call the solve() method to begin the optimization.

3.1. The Problem Environment: Your Interface to the RKO
To make the RKO solver work for your problem, you must create a problem environment class. This class is the bridge between the solver's abstract search methods and the concrete rules of your problem.

It is highly recommended to create a class that implements the structure outlined in the abstract base class RKOEnvAbstract. This ensures all necessary components are correctly defined.

Abstract Environment Template (RKOEnvAbstract)
Python

from abc import ABC, abstractmethod
import numpy as np

class RKOEnvAbstract(ABC):
    """
    Abstract Base Class for creating a problem environment compatible with the RKO solver.
    Inherit from this class and implement all abstract methods and required attributes.
    """
    def __init__(self):
        # --- Required Attributes ---
        self.tam_solution: int = 0
        self.max_time: int = 200
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}
        self.instance_name: str = "default_instance"

        # --- Metaheuristic Parameter Configuration ---
        # Define the parameter space for each metaheuristic.
        # - Single-element list for OFFLINE (static) tuning.
        # - Multi-element list for ONLINE (dynamic) tuning with Q-Learning.
        self.BRKGA_parameters: dict = {'p': [100], 'pe': [0.20], 'pm': [0.10], 'rhoe': [0.70]}
        self.SA_parameters: dict = {'SAmax': [50], 'alphaSA': [0.99], 'betaMin': [0.05], 'betaMax': [0.25], 'T0': [10000]}
        # ... (define for ILS, VNS, PSO, GA, LNS)

        # --- Optional Q-Learning Setting ---
        self.save_q_learning_report: bool = False

    @abstractmethod
    def decoder(self, keys: np.ndarray):
        """
        Translates a random-key vector into a feasible solution.
        """
        pass

    @abstractmethod
    def cost(self, solution, final_solution: bool = False) -> float:
        """
        Calculates the objective function value (cost) of a decoded solution.
        The RKO framework MINIMIZES this value.
        """
        pass
Key Components to Implement:
tam_solution (int): The dimensionality of the random-key vector (e.g., the number of cities in a TSP, the number of items in a knapsack problem).

max_time (int): The maximum execution time in seconds for a single run or restart cycle. This is set by the time_total parameter passed to the solve method.

decoder(self, keys) (method): This is the most critical part. You must implement the logic to convert a NumPy array of random keys into a feasible solution for your problem.

cost(self, solution) (method): This method takes the output of your decoder and returns its numerical objective value. Important: The RKO solver is a minimizer. For maximization problems, you must return a negated value (e.g., return -profit).

Parameter Dictionaries (e.g., SA_parameters): Define the parameter spaces for each metaheuristic. For static parameters, use a single-element list (e.g., 'p': [100]). For dynamic tuning with Q-Learning, provide multiple values (e.g., 'p': [50, 100, 200]).

3.2. Verifying Your Environment with check_env
After creating your environment class, use the check_env function to validate its structure. This utility checks for the presence and correct types of all required attributes and methods, preventing runtime errors.

Python

# Assuming check_env is defined as in the previous response
# from your_utils_file import check_env 

# my_env = YourProblemEnv(...)
# if check_env(my_env):
#     # Proceed with the solver
3.3. Instantiating and Running the Solver
Once your environment is ready, the final step is to instantiate and run the RKO solver.

Python

# 1. Import the necessary classes
from RKO_v2 import RKO
from your_problem_env import YourProblemEnv # Your custom environment class

if __name__ == "__main__":
    # 2. Instantiate your problem environment
    my_environment = YourProblemEnv(dataset_name="my_instance")

    # 3. Instantiate the RKO solver
    # - env: Your custom environment object.
    # - print_best: Set to True to print updates when a new best solution is found.
    # - save_directory: Path to a CSV file to log the results of each run.
    rko_solver = RKO(
        env=my_environment,
        print_best=True,
        save_directory="./results/my_problem_results.csv"
    )

    # 4. Execute the solver
    # - time_total: The overall time limit in seconds for the execution.
    # - runs: The number of independent runs to perform.
    # - restart: The fraction of time_total per restart cycle (1.0 means no restarts).
    # - metaheuristics: Set the number of parallel workers for each algorithm.
    final_cost, final_solution, time_to_best = rko_solver.solve(
        time_total=300,
        runs=5,
        restart=1.0,
        brkga=2,  # Run 2 BRKGA workers
        ils=2,    # Run 2 ILS workers
        vns=1     # Run 1 VNS worker
    )

    # 5. Display the final results
    print("\n--- Final Result ---")
    print(f"Best Objective Value Found: {final_cost}")
    print(f"Time to Find Best Solution: {time_to_best}s")
By following this structure, you can adapt the RKO framework to a wide variety of combinatorial optimization problems, leveraging its powerful, parallel search capabilities with minimal problem-specific coding.
