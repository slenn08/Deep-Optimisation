import torch

def QUBOpopulate(name: str, id: int) -> torch.Tensor:
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
    i = 1
    cur_id = 0
    while cur_id < id:
        size = int(x[i])
        entries = int(x[i + 1])
        i += entries * 3 + 2
        cur_id += 1
    size = int(x[i])
    entries = int(x[i + 1])
    Q = torch.zeros((size, size))
    i += 2
    for _ in range(entries):
        m = int(x[i]) - 1
        n = int(x[i+1]) - 1
        q = int(x[i+2])
        Q[m,n] = q
        Q[n,m] = q
        i += 3

    print("Instance has been loaded")
    return Q