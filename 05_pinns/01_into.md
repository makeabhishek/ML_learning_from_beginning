# (0) Fundamentals and powerful tools in Physics  laws
- Interpretability/generalizability
- Parsimony / Simplicity
- Symmetries / Invarainces / Conservation 

# (1) Neural network architecture
How can we either enforce or promote these ideas in our neural network architecture and how can we also discover these ideas? So we want to force our system to adhere to these ideas using constraint optimisation, boundary condition and loss function.

__Architecture Types__
<img width="298" alt="image" src="https://github.com/user-attachments/assets/573bb493-9896-4d4c-9937-56f21afd4637">

## Parametrizing a space of functions
There is a huge variety of classical and deep learning models available in the ML community. At the end of the data, the ML model takes inputs $(X)$ and tries to build some function $(f)$ that predicts an output of interest $(y)$.

$$
\begin{align}
y=f_{\theta}(X)
\end{align}
$$

This function $(f)$ that we are going to learn, here in this case NN. So in the below figure input is $(X)$, the output is $(y)$ and $(\theta)$ are all the parameters that we can tweak, for example, the weights of the NN to tune/fit this function to get a best-fit model i.e, y as a function of X. \
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/da6670e7-2da7-43de-8ea4-732398d10edd">
</p>

In a Sparse Identification of Nonlinear Dynamics (SINDy) model, the outputs that we are trying to predict is time derivative of some system state $\dot{x}$, and the architecture that we are parametrizing is a bunch of polynomials $(f_{\theta}(X))$ and ${\theta}$ are the weights, which combines with function to approximate our dynamics. \
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/78b61caf-fb72-43d1-9047-684f67a4803b">
</p>

In all the architectures, NN's are trying to constrain the space of this function to fit this admissible function $f$, which describes the input-output mapping through a choice of architecture. So, in all the cases the architecture is parameterized by free parameters $\theta$ and we're gonna optimize those parameters and assign some loss function and optimization algorithm to tune the function to fit the data of the inputs and the outputs.

In summary, architecture defines a space of function we are searching over, and we find the function we want by tuning these free parameters $\theta$. These architectures are parameterizing functions and some parameterizations are more useful for some kinds of physics than others. Some of these functions allow me to enforce symmetries, enforce conservation laws, and promote parsimony and simplicity.

## Different types of Architectures
### Turbulence Modeling: Galilean Invariance
<p align="center">
  <img width="278" alt="image" src="image" src="https://github.com/user-attachments/assets/2dce8766-0106-455a-bf25-5b3dd0b633a6">
</p> 

### Residual Network (RESNET)
RESNET is a deep NN with skip connections. This architecture is designed approximately so that the function of the block behaves like an Euler Integrator or numerical integrator. Good for time-stepping data.
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/79a01a88-83e4-4d35-9908-5d8cbcf3d6d6">
</p> 

### U-NET
Good for image segmentation, and super-resolution. It has a structural inductive bias, which tells us that the things we observe in our daily life are multi-scale in space, which is kind of built implicitly in this architecture. This is good for parametrizing things in natural images.
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/fca05d4e-af73-4bc2-9a98-2eedf0fdfcb9">
</p>

### Physics-Informed Neural Networks (PINNs) 
The first physics-informed NN was introduced by `Lagaris, Isaac E., Aristidis Likas, and Dimitrios I. Fotiadis. "Artificial neural networks for solving ordinary and partial differential equations." IEEE transactions on neural networks 9.5 (1998): 987-1000.`

PINNs are a class of deep learning models that incorporate physical laws, expressed as partial differential equations (PDEs), into the neural network training process. 
The idea is to guide the network to learn solutions that are consistent with both the observed data and the governing physical equations.

Here we are crafting a loss function. utilizing the automatic differentiation capability of NN, we can calculate the gradients to update the model.
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/f9cd53f1-6403-425d-aaeb-420f6cc57cf5">
</p>

![image](https://github.com/user-attachments/assets/b83c77be-27cb-4ff8-8541-e85b23bf66da)

### Lagrangian NN
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/3a26c06d-c63b-4836-8f09-e4095472673c">
</p> 

### Deep Operator Network
Often custom architecture uses less data for training.
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/2ab7551b-405f-4591-a791-9087ba81c7e5">
</p> 

### Fourier Neural operator
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/ff3bd5f0-90d4-444b-960c-9f4ea61b336a">
</p> 

### Graph Neural network
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/d132d2cc-ad73-42ed-b2ba-d585214891cf">
</p> 

## Symmetry, Invariance, and Equivariance
We want our models to satisfy some of these properties.

Invariance: Output doesn't change with any transformation.
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/7a031b80-c09d-4445-849d-0e62143e2e1c">
</p> 

Equivariance:
<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/91e5c86b-fb41-46c7-8339-900fee89ecfd">
</p> 

# (2) Loss Function
Designing a loss function to assess the performance of the model.

How can we add a loss to make an NN a physics-informed loss function

Let's assume, we are trying to predict some physical quantity such as wave velocity wavefield, which has components in x,y, and z as u,v, w, respectively and pressure field. This wavefield varies in space (x,y, z) and time (t).

A naive approach to solve such a problem would be to build a big feed-forward NN, where inputs are spatial locations and output fields are the wavefields at these space and time locations. We can do it if we have large data.

<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/b9002d25-5bb0-4bf1-b806-1b08e21c4398">
</p> 

In conventional NN, we have a loss function in which the output of the loss function tries to match with the actual true data (e.g., velocity/pressure) of the data, and minimise it. In PINNS, we add additional loss terms to satisfy governing equations and boundary conditions. So the NN also satisfy the physics.

PINNS add a second loss function, because of the automatic differentiability of these modern machine learning environments, such as PyTorch, Jacks, and TensorFlow. We can take the output quantities of NN (u,v,w, p) and can compute their partial derivatives with respect to the input of NN (space and time). So we can  build  all of these partial derivatives that went into the partial differential equation (PDE) that you think your system should be satisfying. Therefore, the additional extra loss function here essentially says how much is the governing physical equation (the PDE that governs the physics), how accurate is it, and how much is it violated on the data or on some virtual test points based on these purple quantities in the image. 

Now the the downside is that by adding this physics as a term in the loss function is that you're never really going to exactly satisfy that this loss is zero. So an actual physical system an actual should be exactly zero. The loss suggests that term goes to zero but we're always going to have a balance between these loss functions so you're going to get something that's not perfectly physical. But still, it's more physical than if you did not add it in NN.

<p align="center">
  <img width="278" alt="image" src="https://github.com/user-attachments/assets/38f64ac6-34b9-40bd-8239-65d892860d59">
</p> 

## Data loss
$$
\begin{align}
\sum_{Data} ||\hat{u}(X_j) - u(X_j) ||_2^2
\end{align}
$$

## Least squares (L-2 norm)
to get model error
$$
\begin{align}
L= || AX-b||_2 
\end{align}
$$

## Ridge Regression (Tikhonov regularization)
$$
\begin{align}
L= || AX-b||_2 + \alpha||X||_2
\end{align}
$$

## LASSO (Least Absolute Shrinkage and Selection Operator)
L1 norm to promote sparsity, parsimony to penalise complexity
$$
\begin{align}
L= || AX-b||_2 + \lambda||X||_1
\end{align}
$$

## Elastic Net
$$
\begin{align}
L= || AX-b||_2 + \alpha||X||_2 + \lambda||X||_1 \\
\end{align}
$$

## Equivariance
If we know we have symmetry or equivariance in our data, we can add an additional loss function, instead of doubling our data. Adding more data will make it more expensive to train the NN.

$$
\begin{align}
|| y-f(X)||_2 + ||y-f(-X)||_2
\end{align}
$$
$y-f(-X)$ means we have mirror reflection symmetry. However, we can add any kind of transformation can be added to the loss function. $y$ is the output, $X$ is the input and $f()$ is the NN.

<img width="262" alt="image" src="https://github.com/user-attachments/assets/cd10dbc4-8b33-4d75-87b5-eb61436847c4">

# (3) Optimization
we can add constraints in the loss function and then solve it using constraint optimization algorithms (e.g., KKT algorithm).

If we are just adding a loss function, we are not exactly satisfying constraints. however, if we modify the optimization to constraint optimisation, we are exactly satisfying the constraint. We can add submanifold constraints or subspace constraints.
<img width="561" alt="image" src="https://github.com/user-attachments/assets/e922b9ba-1f7c-41c2-8dc8-789b8c2a4900">

We can also use a genetic algorithm, and symbolic regression (eg. PYSR) as an optimization algorithm designed to find the right model.

Sometimes, we need a custom optimization algorithm for custom loss term in the loss function.

------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------

### Key Concepts of PINNs
#### 1. Integration of Physics:
PINNs integrate physical laws directly into the ``loss function``. This is achieved by ``encoding the differential equations as a penalty term`` that the neural network must satisfy, in addition to fitting any available data.

#### 2. Loss Function Components:
__(.) Data Loss__: Traditional loss based on the difference between the network's predictions and the observed data. This loss is used to train the model by updating its parameters to minimize the discrepancy between predictions and true values. 
There are several traditional loss types commonly used, depending on the nature of the problem (e.g., regression, classification). Common Types of Data Loss Functions used in NN are _Mean Squared Error (MSE) Loss,
Mean Absolute Error (MAE) Loss, Binary Cross-Entropy (BCE) Loss, Categorical Cross-Entropy Loss, Huber Loss_. \
&nbsp; &nbsp;   __(.) Choice of Loss Function:__ The choice of loss function depends on several factors

&nbsp; &nbsp; &nbsp; &nbsp; __(.) Nature of the Task:__ Classification vs. regression, binary vs. multi-class.

&nbsp; &nbsp; &nbsp; &nbsp; __(.) Data Characteristics:__ Presence of outliers, skewness, noise.

&nbsp; &nbsp; &nbsp; &nbsp; __(.) Model Objectives:__ Sensitivity to prediction errors, computational efficiency.

__(.) Physics Loss__: The discrepancy between the neural network's predictions and the differential equations representing the underlying physics. 

__(.) Boundary and Initial Condition Loss__: Enforces known conditions at the domain boundaries and initial time, if applicable. 

#### 3. Automatic Differentiation:

PINNs leverage ``automatic differentiation``, which is a feature of many deep learning frameworks like TensorFlow and PyTorch. This allows for the calculation of 
derivatives required by the PDEs and enables the network to be trained to satisfy these equations.

### Summary
The idea is very simple: add the known differential equations directly into the loss function when training the neural network. This is done by sampling a set of input training locations 
(\{x_{j}\}) for 1D problem, (\{x_{j}, y_{j}\}) for 2D problem and passing them through the network. Next _gradients_ of the network’s output with respect to its input are computed at these locations. Finally, the residual of the underlying differential equation is computed using these gradients and added as an extra term in the loss function. Note that the gradients are typically analytically available for most neural networks, and can be easily computed using auto differentiation.

### Advantages:
__(.) Data Efficiency__: By incorporating physical knowledge, PINNs can often learn from fewer data points, as they are constrained by physics.

__(.) Generalization__: PINNs tend to generalize better to unseen conditions within the constraints of the physical model.

__(.) Handling Noisy Data__: PINNs can smooth out noise in data by relying on the underlying physics.

### Applications:
PINNs are used in various fields, including acoustics, fluid dynamics, heat transfer, electromagnetics, structural mechanics, and beyond, where the underlying systems can be described by PDEs.

### How PINNs Work
Here's a high-level overview of how PINNs are typically implemented:

#### 1. Model Architecture:
A neural network (often fully connected) is designed to approximate the solution to the PDE. Inputs to the network are typically the spatial and temporal coordinates, while outputs are the field variables of interest.

#### 2. Training Process:
The network is trained using both observed data and the physics-based loss. During training, automatic differentiation is used to compute the required derivatives of the network output with respect to its inputs, which are then used to evaluate the physics loss.

#### 3. Optimization:
The optimization aims to minimize the combined loss, ensuring that the network predictions fit the observed data while also satisfying the PDEs as closely as possible.


### Example: Solving a Wave Equation
Consider the 2D wave equation:

$\frac{\partial^2u}{\partial t^2} = c^2 (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})$

The PINN would be trained to minimize a loss function of the form: \
$Loss = \alpha \cdot DataLoss + \beta \cdot ||\frac{\partial^2u}{\partial t^2} - c^2 (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})||^2 + \gamma \cdot Boundary Condition loss$

where:
$u(x,y,t)$ is the wave field (e.g., displacement or pressure); $c$ is the speed of wave propagation; $x,y$  are spatial coordinates, and $t$ is time.

#### Physics Loss Formulation
The physics loss is designed to penalize deviations from the wave equation. It can be formulated using the ``automatic differentiation`` capabilities of deep learning frameworks to
compute the necessary derivatives. The loss can be expressed as follows:

__(1) Residual Definition__: First, deine the residual $R(x,y,t)$ of the wave equation: 

$$
\begin{align}
R(x,y,t) = \frac{\partial^2u}{\partial t^2} - c^2 (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})
\end{align}
$$

__(2) Physics Loss Calculation:__ The physics loss is then the mean squared error of the residual over a set of collocation points. This term ensures the solution satisfies the wave equation within the domain. 

$$
\begin{align}
Loss_{Physics} &= \frac{1}{N} \sum_{i=1}^N (R(x_i,y_i,t_i))^2 \\
Loss_{Physics} &= \frac{1}{N} \sum_{i=1}^N (\frac{\partial^2u}{\partial t^2} - c^2 (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}))^2
\end{align}
$$

where $N$ is the number of collocation points.

__(3) Boundary Loss Calculation:__ This term enforces the free boundary condition at the edges of the domain. 

$$
\begin{align}
Loss_{boundary} = \frac{1}{N_{b}} \sum_{i=1}^{N_{b}} ((\frac{\partial {u_{j}}}{\partial {n}})^2)
\end{align}
$$

__(4) Initial Condition Loss:__ If initial conditions are provided, they can also be included to improve convergence.

$$
\begin{align}
Loss_{initial} = \frac{1}{N_{i}} \sum_{i=1}^{N_{i}} (u_{k} (t=0) -  u_{initial,k})^2
\end{align}
$$

__(5) Total Loss:__ The total loss is a weighted sum of the physics loss, boundary loss, and initial condition loss:

$$
\begin{align}
Loss_{total} = \alpha \cdot Loss_{physics} + \beta \cdot Loss_{boundary} + \gamma \cdot Loss_{initial}
\end{align}
$$

where, $\alpha$, $\beta$, $\gamma$ are weights that can be adjusted to balance the different components of the loss.


### Implementation Steps
__(1) Collocation Points:__ Choose a set of collocation points within the domain where the wave equation should be satisfied. These can be randomly sampled or structured grid points.

__(2) Automatic Differentiation:__ Use a deep learning framework to compute the second-order derivatives of the neural network's output $u(x,y,t)$ with respect to time and space. 

__(3) Compute Residual:__ Calculate the residual $R(x,y,t)$ at each collocation point using the derivatives obtained.

__(4) Physics Loss:__ Compute the mean squared error of the residuals across all collocation points to get the physics loss.



__(.) Data Loss__: Ensures the model fits available data.

__(.) Physics Loss__: Penalizes deviations from the wave equation.

__(.) Boundary Condition Loss__: Ensures the model respects given boundary conditions.




------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------
__References:__
1. Brunton, Steven L., and J. Nathan Kutz. Data-driven science and engineering: Machine learning, dynamical systems, and control. Cambridge University Press, 2022.
2. 


