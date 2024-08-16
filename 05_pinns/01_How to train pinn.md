# A gist of Finite Element (FE) or Finite difference Scheme
Before jumping into PINNS, let us review the steps that we take in the FE or FD method to perform any computation.
1. Define the problem
2. Create geometry and mesh
3. Define the physics laws (Governing equations), and boundary conditions to solve the problem.
4. Define the convergence criteria to obtain the correct solution.
5. Solve the problem.

# A gist of Neural Network (NN)
1. Gather clean input data
2. Feed the data to the NN
3. Define NN hyperparameters (# of layers, # of neurons (nodes), activation function)
4. Train the model and get the output

PNNS is a kind of mixture of numerical modelling and neural networks. However, it's a mesh-free method.

# PINN Training Loop
1. __Collocation points:__ Sample boundary/ physics training points
2. __Output:__ Compute Network outputs
3. __Gradient of Output:__ Compute $1^{st}$ and $2^{nd}$ order gradients of network output with respect to network input.
4. __Loss:__ Compute Loss
5. __Gradient of Loss function:__ Compute the gradient of the loss function concerning network parameters in order to update the parameters to learn the solutions.
6. __Optimization:__ Take gradient descent step

Question: How can we compute the gradients (e.g. $\frac{dNN}{dt}$ and $\frac{L}{d\theta}$)





-------------
------------
References:
1. AI in the Sciences and Engineering, ETH Zurich
