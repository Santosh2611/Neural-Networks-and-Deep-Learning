# Importing the necessary libraries 
import pycrfsuite
from sklearn.metrics import classification_report
from nltk.corpus import conll2002

def load_data(language):
    """
    Load the training and test data for a given language from their respective files.

    Args:
    language (str): A string representing the language, e.g. 'esp' for Spanish, 'ned' for Dutch.

    Returns:
    A tuple of lists containing the training and test data, respectively.
    Each list contains sentences, where each sentence is a list of tuples representing words and corresponding named entity tags.
    """
    train_data = conll2002.iob_sents(f'{language}.train')
    test_data = conll2002.iob_sents(f'{language}.testb')
    return train_data, test_data

def extract_features(sentence):
    """
    Extract features from a given sentence.

    Args:
    sentence (list): A list of tuples representing words and corresponding POS tags.

    Returns:
    A list of dicts, where each dict represents the features for a single word in the sentence.
    """
    return [ {'word': word, 'pos': pos, 'shape': shape if shape and len(shape)>0 else 'X'} for word, pos, *shape in sentence ]


def train_model(X_train, y_train, model_filename):
    """
    Train a CRF model using the given training data and save it to a file.

    Args:
    X_train (list): A list of feature sets, where each feature set corresponds to a sentence in the training data.
    y_train (list): A list of named entity tags, where each tag corresponds to a token in the training data.
    model_filename (str): The filename to use for saving the trained model.
    """
    # Initializing a CRF model using python-crfsuite
    model = pycrfsuite.Trainer(verbose=False)

    # Add the features and labels to the model for training
    for xseq, yseq in zip(X_train, y_train):
        # Check if the number of tokens in sentence equals to the number of labels
        if len(xseq) == len(yseq):
            model.append(xseq, yseq)
        else:
            print(f"Skipping invalid record with |x|={len(xseq)} and |y|={len(yseq)}")

    # Setting the training parameters and train the model
    model.set_params({
        'c1': 0.1,
        'c2': 0.01,
        'max_iterations': 50,
    })

    model.train(model_filename)

def evaluate_model(X_test, y_test, model_filename):
    """
    Evaluate a trained CRF model on the given test data.

    Args:
    X_test (list): A list of feature sets, where each feature set corresponds to a sentence in the test data.
    y_test (list): A list of named entity tags, where each tag corresponds to a token in the test data.
    model_filename (str): The filename of the trained model to use.

    Returns:
    A string representing the classification report.
    """
    # Load the trained model
    tagger = pycrfsuite.Tagger()
    tagger.open(model_filename)

    # make a prediction for each sample in X_test
    y_pred = []
    for xseq in X_test:
        if xseq:
            y_pred.append(tagger.tag(xseq))
        else:
            print("Skipping empty sentence.")

    # flatten the true and predicted labels
    y_test_flat = [label for sent_labels in y_test for label in sent_labels]
    y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

    # Ensure that y_test_flat and y_pred_flat have the same length
    if len(y_test_flat) != len(y_pred_flat):
        return "Lengths of y_test_flat and y_pred_flat do not match."
    else:
        return classification_report(y_test_flat, y_pred_flat)

def predict_entities(sentence, model_filename):
    """
    Predict named entities in a given sentence using a trained CRF model.

    Args:
    sentence (list): A list of tuples representing words and corresponding POS tags.
    model_filename (str): The filename of the trained model to use.

    Returns:
    A list of tuples representing words from the input sentence and their predicted entity tags.
    """
    # Load the trained model
    tagger = pycrfsuite.Tagger()
    tagger.open(model_filename)

    features = extract_features(sentence)
    tags = tagger.tag(features)
    return list(zip(sentence, tags))

if __name__ == '__main__':
    # Load data for Spanish language
    train_data, test_data = load_data('esp')

    # Extract features from all sentences in both datasets
    X_train = [extract_features(sentence) for sentence in train_data]
    y_train = [name for sentence in train_data for _, _, name in sentence]
    X_test = [extract_features(sentence) for sentence in test_data]
    y_test = [name for sentence in test_data for _, _, name in sentence]

    # Train the model and save it to a file
    model_filename = 'ner-esp.crfsuite'
    train_model(X_train, y_train, model_filename)

    # Evaluate the model on the test data and print the classification report
    print(evaluate_model(X_test, y_test, model_filename))

    # Test the model on a sample sentence and print the predicted labels
    sample_sentence = [('Mi', 'DET'), ('nombre', 'NOUN'), ('es', 'AUX'), ('Genie', 'PROPN'), ('y', 'CCONJ'), ('soy', 'AUX'), ('un', 'DET'), ('asistente', 'NOUN'), ('virtual', 'ADJ')]
    predicted_labels = predict_entities(sample_sentence, model_filename)
    print(predicted_labels)
