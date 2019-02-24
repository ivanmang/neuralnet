import numpy as np
# from sklearn.metrics import accuracy_score

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
from illustrate import illustrate_results_FM
def evaluate_architecture(predict, actual):

    pass

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    input_dim = 3
    neurons = [16,32,3]
    activations = ["relu","relu","identity"]
    network = MultiLayerNetwork(input_dim, neurons, activations)

    np.random.shuffle(dataset)

    prep = Preprocessor(dataset)

    x = dataset[:, :3]
    y = dataset[:, 3:]

    train_split_idx = int(0.8 * len(x))
    val_split_idx = int(0.1 * len(x))

    x_train = x[:train_split_idx]
    y_train = y[:train_split_idx]
    x_val = x[val_split_idx:train_split_idx]
    y_val = y[val_split_idx:train_split_idx]
    x_test = x[val_split_idx:]
    y_test = y[val_split_idx:]

    x_train_pre = prep.apply(x_train)
    x_val_pre = prep.apply(x_val)

    trainer = Trainer(
        network=network,
        batch_size=8,
        nb_epoch=100,
        learning_rate=0.01,
        loss_fun="mse",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)

    preds = network(x_val_pre).squeeze()
    # print(preds)
    targets = y_val.squeeze()
    # print(targets)
    accuracy = ((preds - targets)**2).mean()
    print("MSE: {}".format(accuracy/len(preds)))

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
