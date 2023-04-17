import numpy as np

# Define the input features and labels for the logic gates
and_features = np.array([[0,0],[0,1],[1,0],[1,1]])
or_features = np.array([[0,0],[0,1],[1,0],[1,1]])
xor_features = np.array([[0,0],[0,1],[1,0],[1,1]])

and_labels = np.array([0,0,0,1])
or_labels = np.array([0,1,1,1])
xor_labels = np.array([0,1,1,0])

# Define the step function that returns 1 if the input is greater than or equal to 0, else 0
def step_fun(sum):
    return np.where(sum >= 0, 1, 0)

# Initialize the weights and bias for the perceptron algorithm
w = np.random.rand(2)
bias = np.random.rand()

# Define the learning rate and the number of epochs for the algorithm
alpha = 0.1
max_epochs = 1000

print("\nWeight: ", w)
print("Bias: ", bias)
print("Alpha: ", alpha)
print("Epoch: ", max_epochs)

# Implement the perceptron algorithm using vectorized operations (instead of a nested loop) - Calculate the predicted values for all input features at once
def perceptron(features, labels, operator):
    global w, bias
    
    for i in range(max_epochs):        
        print("\nEpoch Value: ", i+1)
        
        # sum = features[j][0]*w[0] + features[j][1]*w[1] + bias
        sum = np.dot(features, w) + bias
        
        predicted = step_fun(sum)        
        delta = labels - predicted
        
        # Update weights and bias using the delta rule
        # w[k] += delta * alpha * features[j][k]
        w += alpha * np.dot(features.T, delta)
        
        # More Efficient Stopping Criterion
        bias += alpha * np.sum(delta)
        error = np.mean(np.abs(delta))
        
        for j in range(len(features)):
            print(features[j][0], "", operator, "", features[j][1], "-> Actual:", labels[j], "; Predicted:", predicted[j], " ( w1:", w[0], "& w2:", w[1], ")")
            
        # Terminate the algorithm if all the predicted values are correct
        if error == 0:
            break
        
# Run the perceptron algorithm for different logic gates
while True:
    operator = str(input("\nEnter Logical Operator (and, or, xor, exit): "))
    
    if operator == "and":
        perceptron(and_features, and_labels, operator)
    elif operator == "or":
        perceptron(or_features, or_labels, operator)
    elif operator == "xor":
        perceptron(xor_features, xor_labels, operator)
    
    elif operator == "exit":
        break
    else:
        print("Invalid Input.\nTry Again!")
