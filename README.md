# RKO - RANDOM-KEY OPTIMIZER (PYTHON FRAMEWORK)

## 1. INTRODUCTION TO THE RANDOM-KEY OPTIMIZER (RKO)

The Random-Key Optimizer (RKO) is a versatile and efficient metaheuristic framework designed for a wide range of combinatorial optimization problems. Its core paradigm is the encoding of solutions as vectors of random keys—real numbers uniformly distributed in the interval [0, 1). This representation maps the discrete, and often complex, search space of a combinatorial problem to a continuous n-dimensional unit hypercube.

The primary strength of the RKO framework lies in its modular architecture, which decouples the search algorithms from the problem-specific logic. This is achieved through a problem-specific decoder, a user-defined function that translates a random-key vector into a feasible solution for the target problem.

This design allows for the seamless integration of multiple classic metaheuristics (e.g., Simulated Annealing, Iterated Local Search, Genetic Algorithms) that can operate independently or in parallel. When executed concurrently, these algorithms share high-quality solutions through a common elite solution pool, fostering a collaborative and robust search process.

---

## 2. FRAMEWORK ARCHITECTURE

The Python implementation of RKO is centered around the **RKO class**, which encapsulates all search operators and metaheuristics.

**Core Operators:**
- Shaking: Diversification mechanism to escape local optima by applying controlled perturbations to a solution.  
- RVND (Random Variable Neighborhood Descent): Intensification strategy that systematically explores multiple neighborhood structures (SwapLS, FareyLS, InvertLS, NelderMeadSearch) in a randomized order to find local optima.  
- Blending: A crossover operator for population-based methods that combines two parent solutions to generate offspring.  
- Parallel Execution: The `solve` method orchestrates the parallel execution of multiple metaheuristic workers using Python's multiprocessing library. These workers operate on the same problem instance and share their findings through a thread-safe `SolutionPool`.  

---

## 3. HOW TO USE THE RKO FRAMEWORK

The main workflow consists of three steps:

1. Define the Problem Environment: Create a Python class that contains all the logic specific to your optimization problem.  
2. Instantiate the RKO Solver: Create an instance of the RKO class, providing your custom environment object and a total time limit.  
3. Execute the Solver: Call the `solve()` method to begin the optimization.  

---

### 3.1. THE PROBLEM ENVIRONMENT: YOUR INTERFACE TO THE RKO

To make the RKO solver work for your problem, you must create a problem environment class.  
It is highly recommended to create a class that implements the structure outlined in the abstract base class **RKOEnvAbstract**.  

```python
from abc import ABC, abstractmethod
import numpy as np

class RKOEnvAbstract(ABC):
    """
    Abstract Base Class for creating a problem environment compatible with the RKO solver.
    Inherit from this class and implement all abstract methods and required attributes.
    """
    def __init__(self):
        # Required attributes
        self.tam_solution: int = 0
        self.max_time: int = 200
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}
        self.instance_name: str = "default_instance"

        # Metaheuristic parameter configuration
        self.BRKGA_parameters: dict = {'p': [100], 'pe': [0.20], 'pm': [0.10], 'rhoe': [0.70]}
        self.SA_parameters: dict = {'SAmax': [50], 'alphaSA': [0.99], 'betaMin': [0.05], 'betaMax': [0.25], 'T0': [10000]}

        # Optional Q-Learning setting
        self.save_q_learning_report: bool = False

    @abstractmethod
    def decoder(self, keys: np.ndarray):
        pass

    @abstractmethod
    def cost(self, solution, final_solution: bool = False) -> float:
        pass
