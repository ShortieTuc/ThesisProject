import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import sklearn.neural_network


def train_and_evaluate(X_train, Y_train, X_test, Y_test):
    # Create a model
    model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                                                 learning_rate_init=0.001, power_t=0.5,
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
                                                 verbose=False, warm_start=False, momentum=0.9,
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                 n_iter_no_change=10)
    # Train the model on the whole data set
    model.fit(X_train, Y_train)
    # Save the model (Make sure that the folder exists)
    joblib.dump(model, 'mlp_classifier.jbl')
    # Evaluate on training data
    print('\n-- Training data --')
    predictions = model.predict(X_train)
    accuracy = sklearn.metrics.accuracy_score(Y_train, predictions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_train, predictions))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_train, predictions))
    print('')
    # Evaluate on test data
    print('\n---- Test data ----')
    predictions = model.predict(X_test)
    u = np.count_nonzero(predictions == 1.0)
    print(u)
    exit()
    accuracy = sklearn.metrics.accuracy_score(Y_test, predictions)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_test, predictions))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_test, predictions))


# Plot the classifier
def plot_classifier(X, Y):
    # Load the model
    model = joblib.load('mlp_classifier.jbl')
    # Calculate
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Make predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot diagram
    fig = plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, cmap='ocean', alpha=0.25)
    plt.contour(xx, yy, Z, colors='w', linewidths=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap='Spectral')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig('plots\\mlp_classifier.png')


# Split dataset to features and target
df = pd.read_csv('../Thesis/Datasets/Denver.csv')
dataset_arr = df.values
X = dataset_arr[:, :-1]
Y = dataset_arr[:, -1:]

# Split into train and test data
trainset_len = int(len(dataset_arr) - 672)
X_train = X[:trainset_len]
Y_train = Y[:trainset_len]
X_test = X[trainset_len:]
Y_test = Y[trainset_len:]

# Train and evaluate
train_and_evaluate(X_train, Y_train, X_test, Y_test)
# Plot classifier
#plot_classifier(X, Y)
