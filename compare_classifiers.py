import argparse
from sklearn.preprocessing import OneHotEncoder

from utils import read_data, split_data
from train_svm import train_smv
from train_nn_classifier import train_model
from train_random_forest import train_forest

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the training data')
    parser.add_argument('--kernel', help='Which kernel to use for the svm.', default='rbf')
    parser.add_argument('--epochs', help='Number of training epochs for the Neural Net.', default=100)
    parser.add_argument('--batch_size', help='Batch size for NN training', default=8)
    parser.add_argument('--split_criterion', help='Either use "entropy" or "gini" impurity for random forest splits', default='gini')
    parser.add_argument('--num_estimators', help='Size of random forest', default='50')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()

    #Read data
    _, X, y = read_data(args.data_path)
    X_train, X_val, y_train, y_val = split_data(X, y)

    #Labels need to be encoded for NN
    encoder = OneHotEncoder()
    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_encoded = encoder.transform(y_val.reshape(-1, 1))

    #Fit models
    nn_model = train_model(X_train, y_train_encoded, int(args.epochs), int(args.batch_size))
    rand_forest = train_forest(X_train, y_train, args.split_criterion, int(args.num_estimators))
    svm = train_smv(X_train, y_train, args.kernel)

    #Get validation performance
    val_performance_nn = nn_model.evaluate(X_val, y_val_encoded)
    val_performance_forest = rand_forest.score(X_val, y_val)
    val_performance_svm = svm.score(X_val, y_val)

    print("Validation performance for NN model was {} (mean accuracy)".format(val_performance_nn[1]))
    print("Validation performance for random forest was {} (mean accuracy)".format(val_performance_forest))
    print("Validation performance for SVM was {} (mean accuracy)".format(val_performance_svm))

if __name__ == '__main__':
    main()
