import argparse

from sklearn.ensemble import RandomForestClassifier

from utils import read_data

def train_forest(data, labels, criterion, n_estimators):
    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    classifier.fit(data, labels)
    return classifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the training data')
    parser.add_argument('--split_criterion', help='Either use "entropy" or "gini" impurity for splits', default='gini')
    parser.add_argument('--num_estimators', help='Size of forest', default='10')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()
    _, X, y, _ = read_data(args.data_path)
    tr_data, val_data, tr_labels, val_labels = split_data(X, y)
    model = train_model(tr_data, tr_labels, args.split_criterion, int(args.num_estimators), oob_score=True)
    performance = model.score(val_data, val_labels)
    print("Performance score was {}".format(performance))
    if args.model_path:
        model.save(args.model_path)

if __name__ == '__main__':
    main()
