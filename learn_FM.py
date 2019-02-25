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

def evaluate_architecture(trainer, x_val_pre, y_val):

    preds = trainer.network.forward(x_val_pre)
    targets = y_val
    mse = r2_score(targets, preds)
    print("Mean squared error regression loss: {}".format(mse))



def main():

    # Load data
    dat = np.loadtxt("FM_dataset.dat")
    # Shuffle data
    np.random.shuffle(dat)

    # Preprocess input data
    prep_input = Preprocessor(dat)
    x = dat[:, :3]
    y = dat[:, 3:]

    train_split_idx = int(0.8 * len(x))

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    # Construct neural network
    neurons = [300, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(3, neurons, activations)

    trainer = Trainer(
        network=net,
        batch_size=50,
        nb_epoch=200,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True)

    trainer.train(x_train_pre, y_train)

    evaluate_architecture(trainer, x_val_pre, y_val)

    illustrate_results_FM(net, prep_input)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



if __name__ == "__main__":
    main()
