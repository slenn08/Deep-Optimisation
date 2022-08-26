from random import choice
from typing import List
import numpy as np
from abc import ABC, abstractmethod
import random

from . import QUBO_populate_function
from . import MKP_populate_function as mkp

class OptimisationProblem(ABC):
    """
    Abstract class to represent an optimisation problem. Any problem to solve via DO should
    subclass from it.
    """
    def __init__(self):
        """
        Constructor method for OptimisationProblem.
        """
        self.max_fitness = float('inf')

    @abstractmethod
    def fitness(self, x: np.ndarray) -> float:
        """
        Calculates the fitness of a solution.

        Args:
            x: numpy.ndarray
                The solution that will have its fitness calculated.
        
        Returns:
            The fitness of the solution.
        """
        pass

    @abstractmethod
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Determines whether a given solution violates any constraints on the problem.

        Args:
            x: numpy.ndarray
                The solution which will be tested.

        Returns:
            True if the solution is valid and False if the solution violates any constraints.
        """
        pass

    @abstractmethod
    def random_solution(self) -> np.ndarray:
        """
        Generates a solution to the problem.

        Returns:
            The random solution.
        """
        pass


class ECProblem(OptimisationProblem):
    def __init__(self, size: int, compression: str, environment: str, linkages: List[int]=None):
        """
        Construtor for a problem with an environment and a compression mappings, as introduced in 
        "Deep Optimisation: Learning and Searching in Deep Representations of Combinatorial
        Optimisation Problems", Jamie Caldwell. 

        Args:   
            size: int
                The size of the problem.
            compression: str
                The compression mapping being used. Should be one of "nov", "ov", "ndov", "npov".
            environment: str
                The environment being used. Should be one of "gc", "hgc", "rs".
            linkages: List[int]
                Specified the linkages between layers of the tree in the HGC environment mapping.
        """
        super().__init__()
        compression = compression.lower()
        environment = environment.lower()
        self.max_fitness = 0
        self.size = size
        compression_func = None
        partial_solutions = [[[-1,-1,-1,-1],[-1,1,-1,1],[1,-1,1,-1],[1,1,1,1]],
                             [[-1,-1,-1,-1],[-1,1,-1,1],[1,-1,-1,1],[1,1,1,1]],
                             [[-1,-1,-1,1],[-1,-1,1,-1],[-1,1,-1,-1],[1,-1,-1,-1]],
                             [[1,1,1,1],[-1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1]]] 
        if compression == "nov":
            compression_func = lambda x: compression_mapping(x, *partial_solutions[0])
        elif compression == "ov":
            compression_func = lambda x: compression_mapping(x, *partial_solutions[1])
        elif compression == "ndov":
            compression_func = lambda x: compression_mapping(x, *partial_solutions[2])
        elif compression == "npov":
            compression_func = lambda x: compression_mapping(x, *partial_solutions[3])
        else:
            raise Exception("Compression mapping not recognised")     

        self.calc_fitness = None
        if environment == "gc":
            self.calc_fitness = lambda x : GC(x, compression_func)
            self.max_fitness = size/2
        elif environment == "hgc":
            if linkages is None:
                linkages = generate_linkage(size)
            self.calc_fitness = lambda x : HGC(x, compression_func, linkages)
            self.max_fitness = (size * 3 / 4) - 1
        elif environment == "rs":
            up = up_matrix(size//4)
            self.calc_fitness = lambda x : RS(x, compression_func, up)
            self.max_fitness = size * (11/8)
        else:
            raise Exception("Environment mapping not recognised")
        
    def fitness(self, x: np.ndarray) -> float:
        """
        Calculates the fitness of the solution given an environment and compression mapping.

        Args:
            x: numpy.ndarray
                The solution that will have its fitness calculated.
        
        Returns:
            The fitness of the solution.
        """
        return self.calc_fitness(x)
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Determines whether a given solution violates any constraints on the problem.

        Args:
            x: numpy.ndarray
                The solution which will be tested.

        Returns:
            This always returns true as there are no constraints on EC problems.
        """
        return True
    def random_solution(self) -> np.ndarray:
        """
        Generates a solution to the problem.

        Returns:
            A random combination of 1s and -1s.
        """
        return np.array([choice([-1,1]) for _ in range(self.size)])

def Fr(R: np.ndarray) -> float:
    """
    Calculates the fitness term that is the number of partial solutions in a solution for EC
    problems.

    Args:
        R: np.ndarray
            The modules of the solution, in shape (W/M, M), where W is the size of the solution and
            M is the size of each module.
    
    Returns:
        The number of partial solutions in the problem.
    """
    total = 0
    for module in R:
        if None not in module:
            total += 1
    return total

def compress(x: np.ndarray, compression) -> np.ndarray:
    """
    Compresses the modules of a solution using its compression mapping.

    Args:
        x: np.nd_array
            The solution to be compressed.
        compression:
            The compression function to be used.
    
    Returns:
        The solution after it has been compressed.
    """
    return np.array([compression(m) for m in x.reshape((x.shape[0]//4,4))])

def compression_mapping(m: np.ndarray, ps1: List[int], ps2: List[int], ps3: List[int], 
                ps4: List[int]) -> np.ndarray:
    """
    Defines the mapping specified by the compression.

    Args:
        m: np.ndarray
            The module that is being compressed.
        ps1, ps2, ps3, ps4: List[int]
            The partial solutions to the compression mapping.
    
    Returns:
        The result to the compression.
    """
    if np.all(m == ps1): return np.array([-1,-1])
    if np.all(m == ps2): return np.array([-1,1])
    if np.all(m == ps3): return np.array([1,-1])
    if np.all(m == ps4): return np.array([1,1])
    else: return np.array([None, None])


def GC(x: np.ndarray, compression):
    """
    Calculates the fitness of a solution in the GC environment.

    Args:
        x: numpy.ndarray
            The solution that will have its fitness calculated. 
        compression:
            The compression mapping being used.
        
        Returns:
            The fitness of the solution.
    """
    # max = size/2
    R = compress(x, compression)
    # fr=size/4 at most
    fr = Fr(R)
    # m = size/4
    m = len(R)
    Hr1 = 0
    Hr2 = 0
    null_present_r1 = False
    null_present_r2 = False
    for r in R:
        if r[0] == 1:
            Hr1 += 1
        if r[1] == 1:
            Hr2 += 1
        if r[0] is None:
            null_present_r1 = True
        if r[1] is None:
            null_present_r2 = True
    r1_score = 0 if null_present_r1 else abs(Hr1 - (m/2))
    r2_score = 0 if null_present_r2 else abs(Hr2 - (m/2))
    return fr + r1_score + r2_score

def A(r1, r2):
    if r1 == r2:
        return r1
    else:
        return None

def generate_linkage(size: int) -> List[int]:
    """
    Generates the random linkage between layers of the HGC tree.

    Args:
        size: int
            The problem size.
    
    Returns:
        The linkages defined as a list.
    """
    linkages = []
    while size > 2:
        size = size // 2
        linkage = [i for i in range(size)]
        random.shuffle(linkage)
        linkages.append(linkage)
    return linkages

def HGC(s: np.ndarray, compression, linkages: List[int]):
    """
    Calculates the fitness of a solution in the HGC environment.

    Args:
        x: numpy.ndarray
            The solution that will have its fitness calculated. 
        compression:
            The compression mapping being used.
        linkages: List[int]
            The linkages between the layers of the tree in HGC.
        
        Returns:
            The fitness of the solution.
    """
    R = compress(s, compression)
    total = Fr(R)
    # Flatten list
    R = R.flatten()
    for linkage in linkages:
        new_R = []
        for i in range(0, len(linkage), 2):
            new_r = A(R[linkage[i]], R[linkage[i+1]])
            if new_r is not None:
                total += 1
            new_R.append(new_r)
        R = new_R
    return total

def up_matrix(l: int) -> np.ndarray:
    """
    Calculates the 'swiss roll' search space needed for the RS environment. This is a 
    matrix of low matrix with a path from (m/2, m) to (m,m) with a monotonically increasing
    fitness along it.

    Args:
        l: int
            The length of the matrix.
        
    Returns:
        The UP matrix used in the RS environment.
    """
    # Generate base for the matrix (without the path)
    x = [list(range(0, x+1)) + [x for _ in range(x+1, l+1)] for x in range(l//2 + 1)]
    y = [list(range(0, x+1)) + [x for _ in range(x+1, l+1)] for x in range(l//2-1, -1, -1)]

    x = x + y

    # Generate the path from (m, m/2) to (m,m)
    counter = l//2
    for i in range(l//2, l):
        x[l//2][i] = counter
        counter += 1
    for i in range(l//2, 0, -1):
        x[i][l] = counter
        counter += 1
    for i in range(l, 0, -1):
        x[0][i] = counter
        counter += 1
    for i in range(0, l):
        x[i][0] = counter
        counter += 1
    for i in range(0, l+1):
        x[l][i] = counter
        counter += 1
    return np.array(x)

def RS(s: np.ndarray, compression, up: np.ndarray) -> float:
    """
    Calculates the fitness of a solution in the GC environment.

    Args:
        x: numpy.ndarray
            The solution that will have its fitness calculated. 
        compression:
            The compression mapping being used.
        up: numpy.ndarray
            The matrix used to calculate where on the path the current solution is.
        
        Returns:
            The fitness of the solution.
    """
    R = compress(s, compression)
    total = Fr(R)
    x = 0
    y = 0
    null_present = False
    for r in R:
        if r[0] == 1:
            x += 1
        if r[1] == 1:
            y += 1
        if None in r:
            null_present = True
    if not null_present:
        total += up[y][x]
    return total

class MKP(OptimisationProblem):
    """
    Class to implement the Multi Dimensional Knapsack Problem.
    """
    def __init__(self, file: str, max_fitness_file: str, id: int):
        """
        Constructor method for MKP. Loads in the weight matrix and constraints matrix for a 
        particular instance of the MKP.

        Args:
            file: str
                The file to extract an MKP instance from.
            max_fitness_file: str
                The file containing the maximum known fitnesses of all the problem instances.
            id: int
                The problem instance ID.
        """
        # c = item values
        # A = dimensions x items
        # # Each row is a dimension
        # # Each column is an item
        # b = knapsack size in each dimension
        self.c, self.A, self.b = mkp.MKPpopulate(file, id)
        self.utility = self.get_utility_order()
        self.max_fitness = mkp.MKPFitness(max_fitness_file, id)
        print(self.max_fitness)

    def fitness(self, x: np.ndarray) -> float:
        """
        Calculate the fitness of any assignment of items.

        Args:
            x: numpy.ndarray
                The solution to have its fitness calculated, where each element is a '1' to 
                represent a 1 and a '-1' to represent a 0.
        
        Returns:
            The fitness.
        """
        if self.is_valid(x):
            # Convert solution from 1s and -1s to 1s and 0s
            x = (x + 1) / 2
            return self.c.dot(x)
        else:
             return 0
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Determines whether a given solution violates any constraints on the problem.

        Args:
            x: numpy.ndarray
                The solution which will be tested.

        Returns:
            True if the solution is valid and False if the solution violates any constraints.
        """
        # Convert from -1 and +1 to 0 and 1
        x = (x + 1) / 2
        return np.all(self.A.dot(x) <= self.b)
    def random_solution(self) -> np.ndarray:
        """
        Generates an empty knapsack.

        Returns:
            A numpy array containing all -1s.
        """
        return np.full(len(self.c), -1)
    
    def get_utility_order(self) -> np.ndarray:
        """
        Calculates the order of the items in terms of their utility.

        Returns:
            A numpy array of the items indices in order of utility increasing.
        """
        # Sum columns to get total weight of each item
        total_weight = self.A.sum(axis=0)
        utility = self.c / total_weight
        # Return indices of items in order of utility ascending
        return np.argsort(utility)
    
    def repair(self, s: np.ndarray) -> np.ndarray:
        """
        Repairs an invalid solution to a good nearby valid solution.

        Args:
            s: np.ndarray   
                The solution to be repaired.
        
        Returns:
            The repaired solution.
        """
        if self.is_valid(s):
            return s
        valid = False
        item_i = 0
        # Remove items with low utility until feasible
        while not valid:
            if item_i >= 100:
                print(s)
                quit()
            i = self.utility[item_i]
            if s[i] == 1:
                s[i] = -1
                valid = self.is_valid(s)
            item_i += 1
        # Start adding items starting from highest utility if feasible
        for i in self.utility[::-1]:
            if s[i] == -1:
                s[i] == 1
                if not self.is_valid(s):
                    s[i] = -1
        return s


class QUBO(OptimisationProblem):
    def __init__(self, file: str, id: int):
        self.Q = QUBO_populate_function.QUBOpopulate(file, id)
        super().__init__()
    
    def fitness(self, x: np.ndarray) -> float:
        x = (x + 1) / 2
        return x.dot(self.Q.dot(x))
    
    def is_valid(self, x: np.ndarray) -> bool:
        return True
    
    def random_solution(self) -> np.ndarray:
        return np.random.randint(0,2,self.Q.shape[0])*2 -1