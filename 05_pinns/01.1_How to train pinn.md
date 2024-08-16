# PINN Training Loop
1. __Collocation points:__ Sample boundary/ physics training points
2. __Output:__ Compute Network outputs
3. __Gradient of Output:__ Compute $1^{st}$ and $2^{nd}$ order gradients of network output with respect to network input.
4. __Loss:__ Compute Loss
5. __Gradient of Loss function:__ Compute the gradient of the loss function with respect to network parameters in order to update the parameters to learn the solutions.
6. __Optimization:__ Take gradient descent step

Question: How can we compute the gradients (e.g. $\frac{dNN}{dt}$ and $\frac{L}{d\theta}$)





-------------
------------
References:
1. AI in the Sciences and Engineering, ETH Zurich
