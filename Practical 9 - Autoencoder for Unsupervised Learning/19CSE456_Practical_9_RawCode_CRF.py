from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
import numpy as np
import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def parse_xml(file_path):
    """
    Read data file and parse the XML.
    """
    with codecs.open(file_path, "r", "utf-8") as infile:
        soup = bs(infile.read(), "html5lib")

    for elem in soup.find_all("document"):
        texts = []

        # Loop through each child of the element under "textwithnamedentities"
        for c in elem.find("textwithnamedentities").children:
            if isinstance(c, Tag):
                if c.name == "namedentityintext":
                    label = "N"  # part of a named entity
                else:
                    label = "I"  # irrelevant word
                for w in c.text.split(" "):
                    if len(w) > 0:
                        texts.append((w, label))
            else:
                # Append the text directly to texts list
                txt = str(c).strip()
                if len(txt) > 0:
                    texts.append((txt, "I"))
        doc = np.array(texts)
        features = np.array(extract_features(doc))
        labels = np.array(get_labels(doc))
        yield doc, features, labels


def word2features(doc, i):
    """
    Extract features for the given word in a document.
    """
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        f'word.lower={word.lower()}',
        f'word[-3:]={word[-3:]}',
        f'word[-2:]={word[-2:]}',
        f'word.isupper={word.isupper()}',
        f'word.istitle={word.istitle()}',
        f'word.isdigit={word.isdigit()}',
        f'postag={postag}'
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            f'-1:word.lower={word1.lower()}',
            f'-1:word.istitle={word1.istitle()}',
            f'-1:word.isupper={word1.isupper()}',
            f'-1:word.isdigit={word1.isdigit()}',
            f'-1:postag={postag1}'
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            f'+1:word.lower={word1.lower()}',
            f'+1:word.istitle={word1.istitle()}',
            f'+1:word.isupper={word1.isupper()}',
            f'+1:word.isdigit={word1.isdigit()}',
            f'+1:postag={postag1}'
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features


def extract_features(doc):
    """
    Extract features from each word in a document.
    """
    return [word2features(doc, i) for i in range(len(doc))]


def get_labels(doc):
    """
    Generate the list of labels for each document.
    """
    return [label for (token, label) in doc]


if __name__ == "__main__":
    # Set the file path to the XML data file
    file_path = "/content/sample_data/reuters.xml"

    # Parse the XML and extract data
    docs = list(parse_xml(file_path))
    print(docs[0])

    # Perform POS tagging on the tokens in the documents
    nltk.download('averaged_perceptron_tagger')
    tagged_data = []
    for doc, features, labels in docs:
        tokens = doc[:, 0]
        tagged = nltk.pos_tag(tokens)
        postags = np.array([pos for word, pos in tagged])
        tagged_data.append(np.column_stack((doc, postags, labels)))
    print(tagged_data[0])

    # Separate data into training set and test set
    X = np.array([doc[:, :2] for doc, _, _ in tagged_data])
    y = np.array([doc[:, 2] for doc, _, _ in tagged_data])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a CRF model using pycrfsuite
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(extract_features(np.column_stack((xseq, yseq))), yseq)
    trainer.set_params({
        'c1': 0.1,
        'c2': 0.01,
        'max_iterations': 200,
        'feature.possible_transitions': True
    })
    trainer.train('crf.model')

    # Test the trained model on the test set
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    y_pred = [tagger.tag(extract_features(xseq)) for xseq in X_test]
    i = 12
    for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in extract_features(X_test[i])]):
        print(f"{y} ({x})")

    # Evaluate model using classification_report
    labels = {}
    for doc in y_train:
        for label in doc:
            if label not in labels:
                labels[label] = len(labels)
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])
    target_names = sorted(labels, key=labels.get)
    class_rep = classification_report(truths, predictions, target_names=target_names)
    print(class_rep)
