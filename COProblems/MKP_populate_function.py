import torch

def MKPpopulate(name: str, id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts the raw data from a .txt file and populates the objective function coefficients
    array, the constraints coefficients matrix and the knapsack capacity array.
    
    Arguments:
        name: str
            the name of the .txt file that contains the raw data.
        id: int
            the id of the problem, starting from 0.
        
    returns:
        c - item values array (shape = n)
        A - item weights in each dimension (shape = m * n)
        b - knapsack capacity in each dimension (shape = m)
        Where n is the number of available items and m is the number of dimensions.
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
        num_columns = int(x[i])
        num_rows = int(x[i+1])
        i += num_columns + (num_columns*num_rows) + num_rows + 3
        cur_id += 1
    num_columns = int(x.pop(i))
    num_rows = int(x.pop(i))
    best = int(x.pop(i))
    # Populating Objective Function Coefficients
    c = torch.Tensor([float(x.pop(i)) for _ in range(num_columns)])
    
    # Populating A matrix (size NumRows * NumColumns)
    const_coef = torch.Tensor([float(x.pop(i)) for _ in range(int(num_rows * num_columns))])           
    A = torch.reshape(const_coef, (num_rows, num_columns)) # reshaping the 1-d ConstCoef into A    
        
    # Populating the RHS
    b = torch.Tensor([float(x.pop(i)) for _ in range(int(num_rows))])

    print("This instance has {} items and {} dimensions".format(num_columns, num_rows))
    return (c, A, b)

def MKPFitness(name: str, id: int) -> float:
    """
    Extracts the fitness of a given MKP instance.

    Args:
        name: str
            The name of the file the fitness is being extracted from.
        id: int
            The id of the problem, starting from 0.
    
    Returns:
        The fitness of the given problem.
    """

    file = open(str(name), 'r')
    x = 0
    for i, line in enumerate(file):
        if i == id:
            x = float(line)
            break
    file.close()
    return x