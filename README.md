This project is a reimplementation of Deep Optimisation. Included is the original AE model used
in Jamie's thesis, as well as a shallow version of DO with a VAE to show how different models
may be implemented. 

Inside the "Models" directory is an abstract class which defines how a model for DO should be
implemented, as well as the deep AE model and shallow VAE model. These define behaviours such
as transitioning, varying solutions, etc.

Inside the "COProblems" directory is an abstract class "OptimisationProblem" that defines how
a CO problem should be implemented to be used with this DO implementation. Included are also 
examples of problems, such as the compression/environment problems defined in Jamie's thesis
and also the MKP with examples from Chu and Beasley, 1997.

In the root directory, OptimHandler is an abstract class that details how to implement a handler.
In this case, a handler is a class that handles some of the core components of the DO algorithm
outside of the model itself, such as optimising solutions, training the model, etc. Each type of
model needs its own type of OptimHandler to be able to run the algorithm.

Also included in the root directory is a series of examples. ec_example.py allows the user to run 
DO on the various compression/environment problems to determine if it has the expected performance.
Also included is running the MKP with with the AE and VAE model, which can be used as a base on 
how to run DO on more applied problems with this implementation.