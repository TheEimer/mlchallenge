import argparse

from sklearn.svm import SVC

from utils import read_data

def train_smv(data, labels, kernel):
    classifier = SVC(kernel=kernel)
    classifier.fit(data, labels)
    return classifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the training data')
    parser.add_argument('--kernel', help='Which kernel to use.', default='rbf')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()
    _, X, y = read_data(args.data_path)
    tr_data, val_data, tr_labels, val_labels = split_data(X, y)
    model = train_model(tr_data, tr_labels, args.kernel)
    performance = model.score(val_data, val_labels)
    print("Performance score was {}".format(performance))
    if args.model_path:
        model.save(args.model_path)

if __name__ == '__main__':
    main()
