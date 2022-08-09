import numpy as np

def QUBOpopulate(name: str, id: int) -> np.ndarray:
    """
    This function extracts the raw data from a .txt file and populates the Q matrix of the
    problem instance
    
    Arguments:
        name: str
            the name of the .txt file that contains the raw data.
        id: int
            the id of the problem, starting from 0.
        
    returns:
        Q - an n x n matrix
        Where n is the number of variables
    """
    
    # Opening .txt file to read raw data of an instance
    file = open(str(name), 'r')
    x = []
    for line in file:
        split_line = line.split()
        for i in range(len(split_line)):
            x.append(split_line[i])
    file.close()

    # Define parameters
    num_problems = int(x.pop(0))
    print(num_problems)
    for _ in range(id + 1):
        size, entries = int(x.pop(0)), int(x.pop(0))
        print(size)
        print(entries)
        Q = np.zeros((size, size))
        for _ in range(entries):
            i = int(x.pop(0)) - 1
            j = int(x.pop(0)) - 1
            q = int(x.pop(0))# * (2 if i != j else 1)
            Q[i,j] = q
            Q[j,i] = q

    print("Instance has been loaded")
    return Q