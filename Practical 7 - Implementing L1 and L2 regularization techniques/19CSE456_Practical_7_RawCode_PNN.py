import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def gaussian(x, b):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-.5 * (x / b) ** 2)


def triangle(x, b):
    return np.where(np.logical_and(np.abs(x / b) <= 1, 1 - np.abs(x / b)), 1, 0)


def epanechnikov(x, b):
    return np.where(np.logical_and(np.abs(x / b) <= 1, True), (3 / 4) * (1 - (x / b) ** 2), 0)


def pattern_layer(inp, kernel, sigma, X_train):
    edis = np.linalg.norm(X_train - inp, axis=1)
    k_values = kernel(edis, sigma)
    return k_values.tolist()


def summation_layer(k_values, Y_train, class_counts):
    # Summing up each value for each class and then averaging
    summed = [np.sum(np.array(k_values)[Y_train.values.ravel() == label]) for label in class_counts.index]
    avg_sum = list(summed / Y_train.value_counts())
    return avg_sum


def output_layer(avg_sum, class_counts):
    max_idx = np.argmax(avg_sum)
    label = class_counts.index[max_idx][0]
    return label


def pnn(X_train, Y_train, X_test, kernel, sigma):
    # Initialising variables
    class_counts = Y_train.value_counts()
    labels = []
    # Passing each sample observation
    for s in X_test:
        k_values = pattern_layer(s, kernel, sigma, X_train)
        avg_sum = summation_layer(k_values, Y_train, class_counts)
        label = output_layer(avg_sum, class_counts)
        labels.append(label)
    print('Labels Generated for bandwidth:', sigma)
    return labels


# Load the iris dataset and convert to Pandas dataframe
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])
print(data.head(5))

# Standardise input and split into train and test sets
X = data.drop(columns='target', axis=1)
Y = data[['target']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y.values.ravel(), train_size=0.8, random_state=12)

# Candidate Kernels
kernels = {'Gaussian': gaussian, 'Triangular': triangle, 'Epanechnikov': epanechnikov}
sigmas = [0.05, 0.5, 0.8, 1, 1.2]

results = pd.DataFrame(columns=['Kernel', 'Smoothing Param', 'Accuracy', 'F1-Score'])
for k, k_func in kernels.items():
    for b in sigmas:
        pred = pnn(X_train, pd.DataFrame(Y_train), X_test, k_func, b)
        accuracy = accuracy_score(Y_test, pred)
        f1 = f1_score(Y_test, pred, average='weighted')
        results.loc[len(results.index)] = [k, b, accuracy, f1]

print(results)

plt.rcParams['figure.figsize'] = [10, 5]
plt.subplot(121)
sns.lineplot(y=results['Accuracy'], x=results['Smoothing Param'], hue=results['Kernel'])
plt.title('Accuracy for Different Kernels', loc='right')

plt.subplot(122)
sns.lineplot(y=results['F1-Score'], x=results['Smoothing Param'], hue=results['Kernel'])
plt.title('F1-Score for Different Kernels', loc='left')

plt.show()
