import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM

def evaluate_architecture(trainer, x_train_pre, y_train, x_val_pre, y_val):
    """Returns the mean squared error regression loss of an architecture

    Arguments:
        trainer -- Object that manages the training of a neural network.
        x_train_pre -- preprocessed training input
        y_train -- training target for x_train_pre
        x_val_pre -- preprocessed Validation input
        y_val -- validation target x_val_pre

    """

    trainer.train(x_train_pre, y_train)

    preds = trainer.network.forward(x_val_pre)
    targets = y_val
    mse = r2_score(targets, preds)

    print("Mean squared error regression loss: {}".format(mse))
    return mse


def main():

    # Load data
    input_dim = 3
    dat = np.loadtxt("FM_dataset.dat")

    # Shuffle data
    np.random.shuffle(dat)

    prep_input = Preprocessor(dat)
    x = dat[:, :3]
    y = dat[:, 3:]

    # Split data into training and validation set
    split_idx = int(0.8 * len(x))
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Preprocess input data
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)


    # Construct neural network
    neurons = [300, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    trainer = Trainer(
        network=net,
        batch_size=100,
        nb_epoch=50,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True)

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

    evaluate_architecture(trainer, x_train_pre, y_train, x_val_pre, y_val)
    illustrate_results_FM(net, prep_input)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



if __name__ == "__main__":
    main()
