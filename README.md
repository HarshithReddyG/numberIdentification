# Artificial Neural Network (ANN) Implementation

This project implements an Artificial Neural Network (ANN) entirely from scratch using **NumPy** for numerical computations. The implementation demonstrates the complete process of building, training, and testing an ANN without relying on external machine learning libraries.

---

## Features of the Project

1. **Custom Implementation**:
   - Implements activation functions (`tanh`, `relu`, `softmax`) and their derivatives for forward and backward propagation.
   - Parameter initialization, gradient descent, and backpropagation are manually coded.

2. **Dataset Processing**:
   - Loads image datasets (`train_X.csv`, `train_label.csv`, `test_X.csv`, `test_label.csv`) using `numpy.loadtxt`.
   - Performs preprocessing to ensure compatibility with the ANN structure.
   - Includes data visualization using `matplotlib` for verification.

3. **Complete ANN Pipeline**:
   - Parameter initialization, forward propagation, cost computation, backpropagation, parameter updates, and prediction are implemented step by step.
   - Modular design allows flexibility for experimentation with different architectures.

4. **Mathematical Rigor**:
   - Uses matrix operations for efficient computations.
   - Implements cross-entropy loss for multi-class classification.

---

## File Overview

### 1. **Dataset Files**
- **`train_X.csv`**: Training data inputs (e.g., flattened 28x28 images).
- **`train_label.csv`**: Training data labels (e.g., one-hot encoded labels for classification).
- **`test_X.csv`**: Testing data inputs.
- **`test_label.csv`**: Testing data labels.

### 2. **Code File**
- **`ANN.py`**:
  - The main script containing the entire implementation of the ANN.
  - Includes functions for initialization, forward/backward propagation, and gradient updates.

---

## Step-by-Step Workflow

### 1. **Data Loading and Visualization**
- Loads training and testing datasets using `np.loadtxt()`.
- Visualizes random samples to ensure the data is loaded correctly.

```python
index = random.randrange(0, X_train.shape[1])
plt.imshow(X_train[:, index].reshape(28, 28), cmap='grey')
plt.show()
```

---

### 2. **Activation Functions**
The project implements the following activation functions:

1. **Tanh**:
   - Formula: `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`
   - Derivative: `1 - tanh^2(x)`

2. **ReLU**:
   - Formula: `relu(x) = max(0, x)`
   - Derivative: `1 if x > 0, else 0`

3. **Softmax**:
   - Converts logits into probabilities for multi-class classification:
     ```python
     expX = np.exp(x)
     softmax = expX / np.sum(expX, axis=0)
     ```

---

### 3. **Parameter Initialization**
Random initialization ensures weights are small to avoid gradient explosion. Biases are initialized to zeros.

```python
def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
```

---

### 4. **Forward Propagation**
- **Step 1**: Compute linear combinations for each layer:
  - Hidden Layer: `z1 = w1.dot(X) + b1`
  - Output Layer: `z2 = w2.dot(a1) + b2`

- **Step 2**: Apply activation functions:
  - Hidden Layer: `a1 = tanh(z1)`
  - Output Layer: `a2 = softmax(z2)`

---

### 5. **Cost Function**
The cost function calculates the cross-entropy loss between predicted and actual labels:

```python
def compute_cost(a2, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(a2)) / m
    return np.squeeze(cost)
```

---

### 6. **Backpropagation**
The gradients of weights and biases are computed as follows:
- **Output Layer**:
  - Error term: `dz2 = a2 - Y`
  - Gradients: `dw2 = dz2.dot(a1.T) / m`, `db2 = np.sum(dz2, axis=1, keepdims=True) / m`

- **Hidden Layer**:
  - Error term: `dz1 = w2.T.dot(dz2) * derivative_tanh(z1)`
  - Gradients: `dw1 = dz1.dot(X.T) / m`, `db1 = np.sum(dz1, axis=1, keepdims=True) / m`

---

### 7. **Parameter Update (Gradient Descent)**
Parameters are updated using the gradients:
```python
parameters["w1"] -= learning_rate * grads["dw1"]
parameters["b1"] -= learning_rate * grads["db1"]
parameters["w2"] -= learning_rate * grads["dw2"]
parameters["b2"] -= learning_rate * grads["db2"]
```

---

### 8. **Training Loop**
The model is trained for a specified number of iterations:
- Perform forward propagation.
- Compute cost.
- Perform backpropagation.
- Update parameters.

---

### 9. **Prediction**
The prediction function uses the trained parameters to classify new inputs:
```python
def predict(parameters, X):
    a2, _ = forward_propagation(X, parameters)
    predictions = np.argmax(a2, axis=0)
    return predictions
```

---

## How to Run
1. Ensure the required datasets (`train_X.csv`, `train_label.csv`, `test_X.csv`, `test_label.csv`) are in the same directory.
2. Run the script using Python:
   ```bash
   python ANN.py
   ```
3. Modify parameters (e.g., number of epochs, learning rate) in the script as needed.

---

## Future Enhancements
1. Add support for more layers and activation functions.
2. Implement learning rate schedulers.
3. Optimize performance using vectorized operations.
4. Add functionality to handle larger datasets dynamically.

---

Feel free to modify the code and experiment further!

