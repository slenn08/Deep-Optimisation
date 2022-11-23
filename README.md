# Introduction
This project is a reimplementation of Deep Optimisation. Included is the original AE model used in Jamie Caldwell's thesis [1], as well as a shallow version of DO with a VAE to show how different models may be implemented. For more details on Deep Optimisation see [4], [5], [6], [7].

# Project Structure
Inside the "Models" directory is an abstract class, DOBase, which defines how a model for DO should be implemented, as well as the deep AE model and shallow VAE model. These define behaviours such as transitioning, varying solutions, etc.

Inside the "COProblems" directory is an abstract class "OptimisationProblem" that defines how a CO problem should be implemented to be used with this DO implementation. Included are also  examples of problems, such as the compression/environment problems defined in the thesis and also the MKP with examples from [2] and QUBO examples from [3].  

In the root directory, OptimHandler is an abstract class that details how to implement a handler. A handler is a class that handles some of the core components of the DO algorithm outside of the model itself, such as optimising solutions, training the model, etc. Each type of model needs its own type of OptimHandler to be able to run the algorithm.

# Problem Examples
## Environment - Compression 
In the root directory is a series of examples. ec_example.py allows the user to run  DO on the various compression/environment problems to determine if it has the expected performance. In terms of how these should be run, the HGC and GC environments should not need much changing compared to the current script, whereas the RS environment generally needs a higher population than given 
(up to 7x as much in some cases) and a compression of 0.6. Some of the other problems may also need an increased population for them to be reliably solved. These problems do not support GPU as for the problem sizes given they do not offer an increase in speed compared to running on the CPU

## MKP
Also included is running the MKP with with the AE and VAE model, which can be used as a base on  how to run DO on more applied problems with this implementation. The MKP instances in this  repository are taken from [2] and are 100 in size. As such, I would recommend running this on the cpu as opposed to the GPU. The scripts are called mkp_DO_example.py and mkp_DOVAE_example.py for the AE and VAE model respectively.

## QUBO
QUBO problem instances from [3], ranging from sizes 100, 500, and 1000, are included in this repository. They are easily run by substituting the problem type in the MKP script examples with the QUBO problem type. In this case, it is recommended to use the GPU for problem sizes > 100. Approximate maximum fitnesses can also be found in [3], although for some problems these are known to not be global maxima.

# Prerequisites
The only prerequisites for this project are matplotlib and pytorch. If not already installed, they can be installed by navigating to the root folder of this repository in the command prompt and running `pip install -r requirements.txt`.

# Contact
If there are any issues or questions feel free to contact me via email at sl2g19@soton.ac.uk

# References
[1] Caldwell, Jamie, Robert (2022) Deep optimisation: learning and searching in deep representations of combinatorial optimisation problems. University of Southampton, Doctoral Thesis, 186pp.

[2] Paul C Chu and John E Beasley (1998). A genetic algorithm for the multidimensional knapsack problem. Journal of heuristics, 4(1):63–86

[3] John E Beasley (1999). Heuristic Algorithms for the Unconstrained Binary Quadratic Programming Problem

[4] Caldwell, J., Knowles, J., Thies, C., Kubacki, F., & Watson, R. (2022). Deep Optimisation: Transitioning the Scale of Evolutionary Search by Inducing and Searching in Deep Representations. SN Computer Science, 3(3), 1-26.

[5] Caldwell, J., Knowles, J., Thies, C., Kubacki, F., & Watson, R. (2021, April). Deep optimisation: multi-scale evolution by inducing and searching in deep representations. In International Conference on the Applications of Evolutionary Computation (Part of EvoStar) (pp. 506-521). Springer, Cham.

[6] Caldwell, J. R., Watson, R. A., Thies, C., & Knowles, J. D. (2018). Deep optimisation: Solving combinatorial optimisation problems using deep neural networks. arXiv preprint arXiv:1811.00784.

[7] Caldwell, J. R., & Watson, R. A. (2017, July). How to get more from your model: the role of constructive selection in estimation of distribution algorithms. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (pp. 101-102).
