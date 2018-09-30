import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense

from utils import read_data, split_data


def train_model(data, labels, epochs, batch_size):
    # Define model: Input layer with 32 neurons, softmax output (1 per class, represents probability of class being correct)
    model = Sequential()
    model.add(Dense(16, input_dim=data.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training happens here
    model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the training data')
    parser.add_argument('test_data', help='Path to test data file')
    parser.add_argument(
        '--epochs', help='Number of training epochs', default=1000)
    parser.add_argument(
        '--batch_size', help='Batch size for training', default=8)
    parser.add_argument('--model_path', help='Where to save model to.')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    _, X, y, X_encoder = read_data(args.data_path)
    tr_data, val_data, tr_labels, val_labels = split_data(X, y)
    X = tr_data
    y = tr_labels
    encoder = OneHotEncoder()
    tr_labels = encoder.fit_transform(tr_labels.reshape(-1, 1))
    val_labels = encoder.transform(val_labels.reshape(-1, 1))

    model = train_model(tr_data, tr_labels, int(
        args.epochs), int(args.batch_size))

    #We need to one-hot the test data in the same way as the training data, so we need the same encoder
    IDs, X_test, _, _ = read_data(args.test_data, test=True, encoder=X_encoder)
    pred = model.predict(X_test)
    pred = encoder.inverse_transform(pred)
    predictions = pd.DataFrame(data=np.column_stack((IDs, pred)), columns=['Unfall_ID', 'Unfallschwere'])

    with open('predictions.txt', 'r+') as f:
        predictions.to_csv(f)

    performance = model.evaluate(val_data, val_labels, batch_size=32)
    print("Performance score was {}".format(performance[1]))
    if args.model_path:
        model.save(args.model_path)


if __name__ == '__main__':
    main()
