### Physics-Informed Neural Networks (PINNs) 
PINNs are a class of deep learning models that incorporate physical laws, expressed as partial differential equations (PDEs), into the neural network training process. 
The idea is to guide the network to learn solutions that are consistent with both the observed data and the governing physical equations.

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

__(1) Residual Definition__: First, deine the residual $R(x,y,t)$ of the wave equation: \
$R(x,y,t) = \frac{\partial^2u}{\partial t^2} - c^2 (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2})$

__(2) Physics Loss Calculation:__ The physics loss is then the mean squared error of the residual over a set of collocation points. This term ensures the solution satisfies the wave equation within the domain. \
$Loss_{Physics} = \frac{1}{N} \sum_{i=1}^N (R(x_i,y_i,t_i))^2$, \
$Loss_{Physics} = \frac{1}{N} \sum_{i=1}^N (\frac{\partial^2u}{\partial t^2} - c^2 (\frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2}))^2$ \
where $N$ is the number of collocation points.

__(3) Boundary Loss Calculation:__ This term enforces the free boundary condition at the edges of the domain. \
$Loss_{boundary} = \frac{1}{N_{b}} \sum_{i=1}^{N_{b}} ((\frac{\partial {u_{j}}}{\partial {n}})^2)$

__(4) Initial Condition Loss:__ If initial conditions are provided, they can also be included to improve convergence.
$Loss_{initial} = \frac{1}{N_{i}} \sum_{i=1}^{N_{i}} (u_{k} (t=0) -  u_{initial,k})^2$

__(5) Total Loss:__ The total loss is a weighted sum of the physics loss, boundary loss, and initial condition loss:

$Loss_{total} = \alpha \cdot Loss_{physics} + \beta \cdot Loss_{boundary} + \gamma \cdot Loss_{initial}$ \
where, $\alpha$, $\beta$, $\gamma$ are weights that can be adjusted to balance the different components of the loss.


### Implementation Steps
__(1) Collocation Points:__ Choose a set of collocation points within the domain where the wave equation should be satisfied. These can be randomly sampled or structured grid points.

__(2) Automatic Differentiation:__ Use a deep learning framework to compute the second-order derivatives of the neural network's output $u(x,y,t)$ with respect to time and space. 

__(3) Compute Residual:__ Calculate the residual $R(x,y,t)$ at each collocation point using the derivatives obtained.

__(4) Physics Loss:__ Compute the mean squared error of the residuals across all collocation points to get the physics loss.



__(.) Data Loss__: Ensures the model fits available data.

__(.) Physics Loss__: Penalizes deviations from the wave equation.

__(.) Boundary Condition Loss__: Ensures the model respects given boundary conditions.




