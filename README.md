# RKO-Python
RKO - Random-Key Optimizer (Python Framework)
1. Introduction to the Random-Key Optimizer (RKO)
The Random-Key Optimizer (RKO) is a versatile and efficient metaheuristic framework designed to address complex combinatorial optimization problems. Its core paradigm is the encoding of solutions as vectors of 

random keys—real numbers uniformly distributed in the interval [0, 1). This representation maps the discrete search space of a combinatorial problem to a continuous n-dimensional unit hypercube.


The primary strength of the RKO framework lies in its modular architecture, which decouples the search algorithms from the problem-specific logic. This is achieved through a problem-specific 

decoder, a function that translates a random-key vector into a feasible solution for the target problem.


This design allows for the integration of multiple classic metaheuristics—such as Simulated Annealing, Iterated Local Search, and Biased Random-Key Genetic Algorithms—which can operate independently or in parallel. When executed concurrently, these algorithms share high-quality solutions through a common elite solution pool, fostering a collaborative and robust search process.

2. Using the RKO Python Framework
The framework is architected for modularity and ease of use. The primary workflow involves three main steps:

Define the Problem Environment: Implement a Python class that encapsulates all problem-specific logic, including instance data, the solution decoder, and the cost function.

Instantiate the RKO Solver: Create an instance of the RKO class, providing your custom environment object during initialization.

Execute the Solver: Call the solve() method to initiate the optimization process.

Code Structure
RKO.py: This file contains the main RKO class, which includes the suite of implemented metaheuristics (e.g., BRKGA, SA, ILS, VNS) and the core search operators (shaking, RVND, Blending).

RKOEnvAbstract (Template): An abstract base class is provided to serve as a formal template for creating custom problem environments.

your_problem_env.py: The user-created file where the custom environment class, inheriting from RKOEnvAbstract, is implemented.

3. Instantiating a New Problem Environment
To adapt the RKO framework for a new optimization problem, the user must implement a problem environment class. This class serves as the interface between the abstract search mechanisms of the RKO solver and the concrete constraints and objectives of the problem.

3.1. The RKOEnvAbstract Base Class
It is highly recommended to inherit from the RKOEnvAbstract class to ensure all necessary components are implemented correctly. This abstract class defines the required structure for any compatible environment.

Python

from abc import ABC, abstractmethod
import numpy as np

class RKOEnvAbstract(ABC):
    """
    Abstract Base Class for creating a problem environment compatible with the RKO solver.
    
    To solve a new problem, create a new class that inherits from this one and implement 
    all abstract methods and define all required attributes.
    """
    def __init__(self):
        # --- Required Attributes ---
        self.tam_solution: int = 0
        self.max_time: int = 200
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}
        self.instance_name: str = "default_instance"
        
        # --- Metaheuristic Parameter Configuration ---
        self.BRKGA_parameters: dict = {'p': [100], 'pe': [0.20], 'pm': [0.10], 'rhoe': [0.70]}
        self.SA_parameters: dict = {'SAmax': [50], 'alphaSA': [0.99], 'betaMin': [0.05], 'betaMax': [0.25], 'T0': [10000]}
        self.ILS_parameters: dict = {'betaMin': [0.10], 'betaMax': [0.20]}
        self.VNS_parameters: dict = {'kMax': [5], 'betaMin': [0.05]}
        self.PSO_parameters: dict = {'PSize': [100], 'c1': [2.05], 'c2': [2.05], 'w': [0.73]}
        self.GA_parameters: dict = {'sizePop': [100], 'probCros': [0.98], 'probMut': [0.005]}
        self.LNS_parameters: dict = {'betaMin': [0.10], 'betaMax': [0.30], 'TO': [1000], 'alphaLNS': [0.95]}

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
3.2. Required Attributes and Methods
Your environment class must define the following:

tam_solution (int): The dimensionality of the random-key vector.

max_time (int): The maximum execution time in seconds.

LS_type (str): The local search strategy, either 'Best' or 'First'.

dict_best (dict): A dictionary mapping instance names to their best-known values.

instance_name (str): The name of the current instance.

Parameter Dictionaries: Dictionaries for each metaheuristic (e.g., SA_parameters), specifying the parameter space. A list with a single value indicates static (offline) tuning, while multiple values enable dynamic (online) tuning with Q-Learning.

decoder(self, keys) (method): This is the most critical implementation. It must contain the logic to convert a NumPy array of random keys into a feasible solution specific to your problem domain.

cost(self, solution) (method): This method must take the output of your decoder and return a single floating-point number representing its objective value. Important: The RKO framework is a minimizer. If your problem is one of maximization (e.g., maximizing profit), you must return the negative of the objective value (e.g., return -profit).

3.3. Verifying the Environment with check_env
To ensure your environment class is correctly implemented, use the provided check_env utility function. It will raise errors if any required component is missing or has an incorrect type.

Python

def check_env(env_instance):
    """
    Verifies that a given environment instance correctly implements the RKOEnvAbstract interface.
    """
    # ... (implementation from previous response)
    return True

4. Executing the Solver
Once your environment class is implemented and verified, running the RKO solver is straightforward.

Python

# 1. Import the necessary classes
from RKO_v2 import RKO
from your_problem_env import YourProblemEnv # Replace with your file and class name

if __name__ == "__main__":
    # 2. Instantiate your problem environment
    # Pass any required arguments, such as dataset name or time limit.
    my_environment = YourProblemEnv(dataset='fu', tempo=300)

    # 3. (Recommended) Verify the environment
    if not check_env(my_environment):
        exit()

    # 4. Instantiate the RKO solver
    rko_solver = RKO(my_environment, print_best=True)

    # 5. Execute the solver
    # Specify the number of parallel workers for each metaheuristic.
    final_cost, final_solution, time_to_best = rko_solver.solve(
        time_total=300,
        runs=1,
        restart=1,
        brkga=1, # 1 worker for BRKGA
        ils=1,   # 1 worker for ILS
        vns=1,   # 1 worker for VNS
        sa=1     # 1 worker for SA
    )

    # 6. Display the results
    print("\n--- Final Result ---")
    print(f"Best Objective Value Found: {final_cost}")
    print(f"Time to Best Solution: {time_to_best}s")
