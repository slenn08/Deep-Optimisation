from random import choice
from re import S
import numpy as np
import sys
import torch

import random
sys.path.append(".")
import COProblems.MKP_populate_function as mkp

class OptimizationProblem():
    def fitness(self, x) -> float:
        return NotImplementedError
    def is_valid(self, x) -> bool:
        return NotImplementedError
    def random_solution(self) -> "list[int]":
        return NotImplementedError

class TestProblem(OptimizationProblem):
    def __init__(self):
        self.bit_string = [choice([1,-1]) for _ in range(32)]
    def fitness(self, x):
        return sum(map(lambda a : a[0]*a[1], zip(x,self.bit_string)))
    def is_valid(self, x):
        return True
    def random_solution(self):
        return [choice([1,-1]) for _ in range(32)]

class HTOP(OptimizationProblem):
    def __init__(self, size):
        self.size = size
    def fitness(self,x):
        fitness_val = 0
        while len(x) > 2:
            new_x = []
            for i in range(0,len(x),4):
                chunk = x[i:i+4]
                new_chunk = self.t(chunk)
                fitness_val += self.f(new_chunk)
                new_x += new_chunk
            x = new_x
        return fitness_val
    def f(self, x):
        if None not in x:
            return 1
        else:
            return 0
    def t(self, x):
        if x == [1,-1,-1,-1]: return [-1,-1]
        if x == [-1,1,-1,-1]: return [-1,1]
        if x == [-1,-1,1,-1]: return [1,-1]
        if x == [-1,-1,-1,1]: return [1,1]
        else: return [None, None]
    def is_valid(self, x):
        return True
    def random_solution(self):
        return [choice([1,-1]) for _ in range(self.size)]

class MCParity(OptimizationProblem):
    def fitness(self, x):
        modules = [x[i:i + 4] for i in range(0, len(x), 4)]
        types = [[1,1,-1,1], [1,-1,-1,-1]]
        p = 0.0001
        fitness = 0
        for m in modules:
            fitness += (abs(sum(m))==2)
        for t in types:
            total = 0
            for m in modules:
                if m == t:
                    total += 1
            fitness += p * (total ** 2)
        
        return fitness
            
    def is_valid(self, x):
        return True
    def random_solution(self):
        return [choice([1,-1]) for _ in range(128)]

class ECProblem(OptimizationProblem):
    def __init__(self, size, compression, environment, linkages=None):
        self.max_fitness = 0
        self.size = size
        compression_func = None
        if compression == "nov":
            compression_func = nov_compression
        elif compression == "ov":
            compression_func = ov_compression
        elif compression == "ndov":
            compression_func = ndov_compression
        elif compression == "npov":
            compression_func = npov_compression
        else:
            raise Exception("Compression mapping not recognised")     

        self.calc_fitness = None
        if environment == "gc":
            self.calc_fitness = lambda x : GC(x, compression_func)
            self.max_fitness = size/2
        elif environment == "hgc":
            if linkages is None:
                linkages = generate_linkage(size)
            # self.calc_fitness = lambda x : HGC(x, compression_func)
            self.calc_fitness = lambda x : HGC2(x, compression_func, linkages)
            self.max_fitness = (size * 3 / 4) - 1
        elif environment == "rs":
            up = up_matrix(size//4)
            self.calc_fitness = lambda x : RS(x, compression_func, up)
            self.max_fitness = size/4 + size*(9/8)
        else:
            raise Exception("Environment mapping not recognised")
        
    def fitness(self, x):
        return self.calc_fitness(x)
    def is_valid(self, x):
        return True
    def random_solution(self):
        return [choice([-1,1]) for _ in range(self.size)]

def Fr(R):
    total = 0
    for module in R:
        if module != [None, None]:
            total += 1
    return total

def compress(s, compression):
    return [compression(s[i:i + 4]) for i in range(0, len(s), 4)]

def nov_compression(m):
    if m == [-1,-1,-1,-1]: return [-1,-1]
    if m == [-1,1,-1,1]: return [-1,1]
    if m == [1,-1,1,-1]: return [1,-1]
    if m == [1,1,1,1]: return [1,1]
    else: return [None, None]

def ov_compression(m):
    if m == [-1,-1,-1,-1]: return [-1,-1]
    if m == [-1,1,-1,1]: return [-1,1]
    if m == [1,-1,-1,1]: return [1,-1]
    if m == [1,1,1,1]: return [1,1]
    else: return [None, None]

def ndov_compression(m):
    if m == [-1,-1,-1,1]: return [-1,-1]
    if m == [-1,-1,1,-1]: return [-1,1]
    if m == [-1,1,-1,-1]: return [1,-1]
    if m == [1,-1,-1,-1]: return [1,1]
    else: return [None, None]

def npov_compression(m):
    if m == [1,1,1,1]: return [-1,-1]
    if m == [-1,-1,1,1]: return [-1,1]
    if m == [-1,1,-1,-1]: return [1,-1]
    if m == [1,-1,-1,-1]: return [1,1]
    else: return [None, None]

def GC(s, compression):
    # max = size/2
    R = compress(s, compression)
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

def HGC(s, compression):
    R = compress(s, compression)
    total = Fr(R)
    while len(R) >= 2:
        new_R = []
        for r in R:
            new_r = A(r[0], r[1])
            if new_r is not None:
                total += 1
            new_R.append(new_r)
        R = [new_R[i:i + 2] for i in range(0, len(new_R), 2)]
    r = R[0]
    if A(r[0], r[1]):
        total += 1
    return total

def generate_linkage(size):
    linkages = []
    while size > 2:
        size = size // 2
        linkage = [i for i in range(size)]
        random.shuffle(linkage)
        linkages.append(linkage)
    return linkages
def HGC2(s, compression, linkages):
    R = compress(s, compression)
    total = Fr(R)
    # Flatten list
    R = [i for r in R for i in r]
    for linkage in linkages:
        new_R = []
        for i in range(0, len(linkage), 2):
            new_r = A(R[linkage[i]], R[linkage[i+1]])
            if new_r is not None:
                total += 1
            new_R.append(new_r)
        R = new_R
    return total

def up_matrix(l):
    x = [list(range(0, x+1)) + [x for _ in range(x+1, l+1)] for x in range(l//2 + 1)]
    y = [list(range(0, x+1)) + [x for _ in range(x+1, l+1)] for x in range(l//2-1, -1, -1)]

    x = x + y
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
    return x

def RS(s, compression, up):
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

class MKP(OptimizationProblem):
    def __init__(self, file, id):
        # c = item values
        # A = dimensions x items
        # # Each row is a dimension
        # # Each column is an item
        # b = knapsack size in each dimension
        self.c, self.A, self.b = mkp.MKPpopulate(file, id)
        self.utility = self.get_utility_order()
        self.max_fitness = 0
    def fitness(self, x) -> float:
        if self.is_valid(x):
            x = np.array(x)
            x = (x + 1) / 2
            return self.c.dot(np.array(x))
        else:
             return 0
    def is_valid(self, x) -> bool:
        # Convert from -1 and +1 to 0 and 1
        x = np.array(x)
        x = (x + 1) / 2
        return np.all(self.A.dot(x) <= self.b)
    def random_solution(self):
        return [-1 for _ in range(len(self.c))]
    
    def get_utility_order(self):
        # sum columns
        total_weight = self.A.sum(axis=0)
        utility = self.c / total_weight
        # return indices of items in order of utility ascending
        return np.argsort(utility)
    
    def repair(self, s):
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
                    s[i] = 0
        return s


if __name__=="__main__":
    print(up_matrix(8))
