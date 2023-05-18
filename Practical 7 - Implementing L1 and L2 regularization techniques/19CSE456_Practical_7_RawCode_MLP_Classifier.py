from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_and_split_data(test_size = 0.2):
    '''
    Loads iris dataset and splits it into train and test datasets
    
    Parameters:
    test_size: float (default=0.2)
        This parameter determines the fraction of total data to be used for testing.

    Returns:
    Tuple containing train_data, test_data, train_labels, test_labels
    '''
    
    # Load Iris dataset
    iris = load_iris()

    # Splitting into train and test datasets
    return train_test_split(iris.data, iris.target, test_size=test_size)

def preprocess_data(train_data, test_data):
    '''
    Scales the data using StandardScaler
    
    Parameters:
    train_data: numpy array
        The numpy array contains training data
    test_data: numpy array
        The numpy array contains testing data
        
    Returns:
    Tuple containing scaled_train_data and scaled_test_data
    '''
    
    scaler = StandardScaler()
    scaler.fit(train_data)

    return scaler.transform(train_data), scaler.transform(test_data)

if __name__ == "__main__":
    # Loading and splitting the data
    train_data, test_data, train_labels, test_labels = load_and_split_data()

    # Preprocessing the data
    train_data, test_data = preprocess_data(train_data, test_data)

    # Creating an classifier from the neural network model:
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

    # Fitting training data to our model
    mlp.fit(train_data, train_labels)

    # Predicting with the trained model and calculating accuracy
    predictions_train = mlp.predict(train_data)
    train_accuracy_score = accuracy_score(predictions_train, train_labels)

    predictions_test = mlp.predict(test_data)
    test_accuracy_score = accuracy_score(predictions_test, test_labels)

    # Confusion matrix
    confusion_mat = confusion_matrix(predictions_train, train_labels)

    # Classification report
    cls_report = classification_report(predictions_test, test_labels)

    # Printing the results
    print(f'\nTrain Accuracy Score: {train_accuracy_score:.2f}')
    print(f'Test Accuracy Score: {test_accuracy_score:.2f}')
    print(f'\nConfusion Matrix:\n{confusion_mat}')
    print(f'\nClassification Report:\n{cls_report}')
