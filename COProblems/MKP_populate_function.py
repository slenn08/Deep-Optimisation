# # Multidimensional Knapsack Problem

# Mohammed Alagha, June 2021
# 
# Glasgow, UK

# A function to read MKP data and populate an instance.


from typing import Tuple
import numpy as np

def MKPpopulate(name: str, id: int) -> Tuple:
    # populateMKP

    '''
    This function extracts the raw data from a .txt file and populates the objective function coefficients
    array, the constraints coefficients matrix A and the right hand side b array
    
    Arguments:
        name -- the name of the .txt file that contains the raw data
        
    returns:
        c -- objective function coefficients array (shape = 1 * n)
        A -- constraints coefficients matrix A (shape = m * n)
        b -- right hand side values (shape = 1 * m)
    '''
    
    # Opening .txt file to read raw data of an instance
    file = open(str(name), 'r')
    x = []
    for line in file:
        splitLine = line.split()
        for i in range(len(splitLine)):
            x.append(splitLine[i])
    file.close()

    # Define parameters
    NumProblems = int(x.pop(0))
    for _ in range(id + 1):
        NumColumns, NumRows, BestOF = int(x.pop(0)), int(x.pop(0)), float(x.pop(0))
        print('This instance has %d variables and %d constraints' %(NumColumns, NumRows))

        if BestOF != float(0):
            print('Best known integer objective value for this instance = ', BestOF)
        else:
            print('Best integer objective value for this instance is not indicated')
        
        # Populating Objective Function Coefficients
        c = np.array([float(x.pop(0)) for i in range(NumColumns)])
        
        assert type(c) == np.ndarray
        assert len(c)  == NumColumns
        
        
        # Populating A matrix (size NumRows * NumColumns)
        ConstCoef = np.array([float(x.pop(0)) for i in range(int(NumRows * NumColumns))])    
        
        assert type(ConstCoef) == np.ndarray
        assert len(ConstCoef)  == int(NumRows*NumColumns)
        
        A = np.reshape(ConstCoef, (NumRows, NumColumns)) # reshaping the 1-d ConstCoef into A
        
        assert A.shape == (NumRows, NumColumns)
        
        
        # Populating the RHS
        b = np.array([float(x.pop(0)) for i in range(int(NumRows))])

        assert len(b) == NumRows
        assert type(b) == np.ndarray
        
    return (c, A, b)

#def MKPFitness(self, )