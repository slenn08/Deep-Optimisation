import random

import torch

from COProblems.OptimisationProblem import OptimisationProblem

class ECProblem(OptimisationProblem):
    def __init__(self, size: int, compression: str, environment: str, linkages: list[list[int]]=None):
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
            linkages: list[list[int]]
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
        partial_solutions = [[torch.tensor(x) for x in y] for y in partial_solutions]

        c_list = ["nov", "ov","ndov","npov"]
        compression_i = {c:i for i, c in enumerate(c_list)}
        if compression not in c_list:
            raise Exception("Compression mapping not recognised, please enter one of 'nov', 'ov', 'ndov', 'npov")
        compression_func = lambda x: compression_mapping(x, *partial_solutions[compression_i[compression]])   

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
            raise Exception("Environment mapping not recognised, please enter one of 'gc',' hgc', 'rs'")
        
    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the fitnesses of the solutions given an environment and compression mapping.

        Args:
            x: torch.Tensor
                The solutions that will have their fitnesses calculated.
        
        Returns:
            The fitnesses of the solutions.
        """
        return self.calc_fitness(x)

    def is_valid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determines whether given solutions violate any constraints on the problem.

        Args:
            x: numpy.ndarray
                The solutions which will be tested.

        Returns:
            Always returns a tensor containing trues as there are no constraints on EC problems.
        """
        return torch.full((x.shape[0],), True)
    def random_solution(self, pop_size: int) -> torch.Tensor:
        """
        Generates random solutions to the problem.

        Args:
            pop_size: int
                The size of the population of solutions.

        Returns:
            A tensor of random solutions consisting of a combination of 1s and -1s.
        """
        return torch.randint(0,2,(pop_size, self.size), dtype=torch.float32) * 2 - 1
    
    def repair(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repairs solutions so that they remain valid.

        Args:
            x: torch.Tensor
                The solutions to be repaired.
        
        Returns:
            Returns the population unchanged as there are no constraints.
        """
        return x


def Fr(solutions: torch.Tensor) -> torch.Tensor:
    """
    Calculates the fitness term that is the number of partial solutions in a solution for EC
    problems.

    Args:
        solutions: torch.Tensor
            The solutions in the population.
    
    Returns:
        A fitness tensor where each element i is the number of partial solutions in solution i.
    """
    fitnesses = torch.sum(~((solutions[:,::2] == 0) | (solutions[:,1::2] == 0)), axis=1, dtype=torch.float32)
    return fitnesses

def compression_mapping(s: torch.Tensor, ps1: torch.Tensor, ps2: torch.Tensor, ps3: torch.Tensor, 
                ps4: torch.Tensor) -> torch.Tensor:
    """
    Defines the mapping specified by the compression.

    Args:
        s: torch.Tensor
            The solutions that are being compressed.
        ps1, ps2, ps3, ps4: torch.Tensor
            The partial solutions to the compression mapping.
    
    Returns:
        The result of the compression.
    """
    reshaped_x = s.view(s.shape[0],-1,4)
    compressed = torch.zeros((s.shape[0], s.shape[1]//4, 2))
    matches = (reshaped_x == ps1).all(axis=2)
    compressed[matches] = torch.tensor([-1,-1], dtype=torch.float32)
    matches = (reshaped_x == ps2).all(axis=2)
    compressed[matches] = torch.tensor([-1,1], dtype=torch.float32)
    matches = (reshaped_x == ps3).all(axis=2)
    compressed[matches] = torch.tensor([1,-1], dtype=torch.float32)
    matches = (reshaped_x == ps4).all(axis=2)
    compressed[matches] = torch.tensor([1,1], dtype=torch.float32)

    return compressed.reshape(s.shape[0], s.shape[1]//2)


def GC(solutions: torch.Tensor, compression) -> torch.Tensor:
    """
    Calculates the fitness of a solution in the GC environment.

    Args:
        solutions: torch.Tensor
            The solutions that will have their fitnesses calculated. 
        compression:
            The compression function with specified partial solutions being used.
        
        Returns:
            The fitnesses of the solutions.
    """
    # length = size/2
    compressed = compression(solutions)
    # fr=size/4 at most
    fitnesses = Fr(compressed)
    # m = size/4
    m = compressed.shape[1] / 2

    r1 = compressed[:,::2]
    r2 = compressed[:,1::2]
    r1_score = torch.where(torch.all(r1!=0, dim=1), abs((r1==1).sum(dim=1) - (m/2)), 0) 
    r2_score = torch.where(torch.all(r2!=0, dim=1), abs((r2==1).sum(dim=1) - (m/2)), 0) 
    fitnesses += r1_score + r2_score
    
    return fitnesses

def generate_linkage(size: int) -> list[list[int]]:
    """
    Generates the random linkage between layers of the HGC tree.

    Args:
        size: int
            The problem size.
    
    Returns:
        The linkages defined as a list of lists, where for each list element i links with element i + 1, where i is even.
    """
    linkages = []
    while size > 2:
        size = size // 2
        linkage = [i for i in range(size)]
        random.shuffle(linkage)
        linkages.append(linkage)
    return linkages

def HGC(solutions: torch.Tensor, compression, linkages: list[list[int]]) -> torch.Tensor:
    """
    Calculates the fitness of a solution in the HGC environment.

    Args:
        solutions: torch.Tensor
            The solutions that will have their fitnesses calculated. 
        compression:
            The compression mapping being used.
        linkages: list[ist[int]]
            The linkages between the layers of the tree in HGC.
        
        Returns:
            The fitnesses of the solutions.
    """
    compressed = compression(solutions)
    fitnesses = Fr(compressed)
    for linkage in linkages:
        compared_r1 = compressed[:,linkage[0::2]]
        compared_r2 = compressed[:,linkage[1::2]]
        new_solutions = torch.where(compared_r1 == compared_r2, compared_r1, 0)
        fitnesses += (new_solutions != 0).sum(dim=1)
        compressed = new_solutions
    return fitnesses

def up_matrix(l: int) -> torch.Tensor:
    """
    Calculates the 'swiss roll' search space needed for the RS environment. This is a 
    matrix of small values with a path from (m/2, m) to (m,m) with a monotonically increasing
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
    return torch.tensor(x)

def RS(solutions: torch.Tensor, compression, up: torch.Tensor) -> torch.Tensor:
    """
    Calculates the fitnesses of solutions in the GC environment.

    Args:
        solutions: torch.Tensor
            The solutions that will have their fitnesses calculated. 
        compression:
            The compression mapping being used.
        up: torch.Tensor
            The matrix used to calculate where on the path the current solution is.
        
        Returns:
            The fitnesses of the solutions.
    """
    compressed = compression(solutions)
    fitnesses = Fr(compressed)
    null_present = False

    r1 = compressed[:,::2]
    r2 = compressed[:,1::2]
    xs = (r1 == 1).sum(dim=1)
    ys = (r2 == 1).sum(dim=1)
    up_f = up[ys,xs]
    fitnesses += torch.where(torch.all((r1 != 0) & (r2 != 0)), up_f, 0)
    return fitnesses