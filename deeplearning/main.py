import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_loader import load_data
from utils import log


plt.ion()  # Turn on interactive mode


def hyp_tan(z):
    return np.tanh(z)


def hyp_tan_prime(z):
    """Derivative of tanh"""
    return 1 - np.pow(np.tanh(z), 2)


def softsign(z):
    return z / (1 + np.abs(z))


def softsign_prime(z):
    return 1 / np.pow((np.abs(z) + 1), 2)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1.0 - sigmoid(z))


class Network(object):
    # sizes is number of neurons in each layer
    def __init__(self, sizes, activation=sigmoid, activation_prime=sigmoid_prime):
        self.num_layers = len(sizes)
        self.activation = activation
        self.activation_prime = activation_prime

        # each layer i in sizes has sizes[i] neurons
        # first layer is the input so we skip that
        # each neuron has one bias value
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        # each neuron has multiple weights
        # the number of weights of neurons in layer x is the number of neurons in layer x - 1
        # so we need to initialise x_n neurons, each of which is an array of (x-1)_n elements
        self.weights = [
            np.random.randn(this_layer_n, last_layer_n)
            for this_layer_n, last_layer_n in zip(sizes[1:], sizes[:-1])
        ]

        # initialise gradients for weights and biases to zeros
        # note that this is del (aka the gradient) and not delta
        # we will later convert this to delta (which is actual amount of change in weights and biases)
        # using eta
        self.del_w = [np.zeros(w.shape) for w in self.weights]
        self.del_b = [np.zeros(b.shape) for b in self.biases]
        self.fig = plt.figure()
        # set title to activation name
        # also add layer sizes to title
        self.plot_title = f"Activation: {activation.__name__}, Layers: {sizes}"
        self.summary_stats = []

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y

    def feedforward(self, input):
        # go through all layers in our network
        # and calculate activations for each layer
        # which is basically running the self.activation function for each layer
        # after doing a dot product of w * a and adding bias b to each neuron
        a = input
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # log(f"Calculating for layer: {i + 2}\n")
            # log(f"Inputs of this layer: \n{a}\n")
            # log(f"Weights of this layer size:\n{w}\n")
            # for first layer a will be shape of input
            # and if second layer's neurons are n in number
            a = self.activation(np.dot(w, a) + b)
            # log(f"Final calculated Activations of this layer: \n{a}\n")

        return a

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x

        # the first layer's activations are just the input
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # if i == 0:
            #     print("Weights shape: ", w.shape)
            #     print("Biases shape: ", b.shape)
            #     print("Activations: ", len(activation))
            #     print("\n---\n")
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)

        summary_stats = []
        # calculate the mean and standard deviations of the activations of each layer
        for i, layer_activations in enumerate(activations):
            if i == 0 or i == len(activations) - 1:
                # skip input and output layers
                continue
            mean = np.mean(layer_activations)
            dev = np.std(layer_activations)
            summary_stats.append({"mean": mean, "dev": dev, "layer": i + 1})

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * self.activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_w, nabla_b, summary_stats

    def zero_grad(self):
        """
        Set all gradients back to zeros
        """
        self.del_w = [np.zeros(w.shape) for w in self.weights]
        self.del_b = [np.zeros(b.shape) for b in self.biases]

    def update_weights(self, eta, batch_num):
        """
        Update the weights and biases of the network
        Using the given gradients
        """
        # tqdm.write(f"Batch number: {batch_num}, Updating weights")
        # print("Old weight shapes", len(self.weights[0]))
        # print("Old biases shapes", len(self.biases[0][0]))
        # print("Old del_w shapes", len(self.del_w[0][0]))
        # print("Old del_b shapes", len(self.del_b[0][0]))

        # pdb.set_trace()
        # we will multiply the gradients by eta
        # to get the DELTA aka the factor by which we want to change the weights and biases
        self.weights = [w - (eta * dw) for w, dw in zip(self.weights, self.del_w)]
        self.biases = [b - (eta * db) for b, db in zip(self.biases, self.del_b)]

        # pdb.set_trace()

        # print("New weight shapes", len(self.weights[0]))
        # print("New biases shapes", len(self.biases[0][0]))
        # print("New del_w shapes", len(self.del_w[0][0]))
        # print("New del_b shapes", len(self.del_b[0][0]))

    def update_gradients(self, batch, epoch_end=False):
        """
        Compute the gradients within a batch
        """

        batch_size = len(batch)
        # we will go through each batch, and sum the gradients for each batch entry
        for input, label in batch:
            # call the back prop method
            # which passes this input through the network
            # and returns the gradients of the loss
            # computed based on these inputs
            this_item_del_w, this_item_del_b, stats = self.backprop(input, label)
            # keep adding these gradients (divided by batch size) to our global del_w and del_b
            self.del_w = [
                dw + (this_item_dw / batch_size)
                for dw, this_item_dw in zip(self.del_w, this_item_del_w)
            ]
            self.del_b = [
                db + (this_item_db / batch_size)
                for db, this_item_db in zip(self.del_b, this_item_del_b)
            ]

        self.summary_stats.append(stats)

        plt.clf()

        for layer_num in range(
            len(stats)
        ):  # Assuming 'stats' contains entries for each layer
            # Debugging: Print the mean values being plotted for this layer
            mean_values = [
                batch_stats[layer_num]["mean"] for batch_stats in self.summary_stats
            ]

            plt.plot(
                [
                    batch_num for batch_num in range(len(self.summary_stats))
                ],  # X-axis: batch numbers
                mean_values,  # Y-axis: mean activations per layer per batch
                label=f"Layer {layer_num + 1}",
            )

        plt.legend()
        plt.title(self.plot_title)
        plt.draw()  # Update the plot with the new data
        plt.pause(0.1)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(
        self,
        training_data,
        batch_size,
        epochs,
        gradient_accumulation_steps,
        eta,
        shuffle=False,
        test_data=None,
    ):
        """
        Divide training data into batch_sizes
        Go through each mini_batch
        Every gradient_accumulation_steps number of mini_batches, update the gradients
        update after every epoch
        """
        # if training_data is None:
        #     raise Exception("Training data must be provided")

        # if type(training_data) != np.ndarray:
        #     raise Exception("Training data must be a numpy array")

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        log(f"Len training data: {n}")
        log(f"Batch size: {batch_size}")
        log(f"Num batches: {np.ceil(n / batch_size)}")

        for epoch_idx in range(epochs):
            # divide the training data into batch_size arrays
            if shuffle:
                np.random.shuffle(training_data)

            batches = [
                training_data[k : k + batch_size] for k in np.arange(0, n, batch_size)
            ]

            log(f"Starting epoch: {epoch_idx + 1}")

            with tqdm(total=len(batches), position=0, leave=True) as pbar:
                for batch_idx, batch in enumerate(batches):
                    # log(f"Batch: {batch_idx + 1} / {len(batches)}", indent=1)
                    # update gradients with this batch's items
                    self.update_gradients(batch=batch)

                    if (
                        (batch_idx + 1) % gradient_accumulation_steps
                    ) == 0 or batch_idx == len(batches) - 1:
                        # update weights and biases
                        self.update_weights(eta, batch_idx + 1)
                        # set gradients back to zero
                        self.zero_grad()

                    pbar.update(1)

            # test after every epoch
            if test_data:
                num_correct = self.evaluate(test_data)
                log(
                    f"Eval result: {num_correct} / {n_test}",
                    color_code="g",
                )
            else:
                log(f"Finished epoch: {epoch_idx + 1}")

        # save plot
        # get the name of the activation function
        activation_name = self.activation.__name__
        plt.savefig(f"activations-{activation_name}.png")


sizes = [784, 40, 40, 10]


nets = [
    {
        "sizes": sizes,
        "activation": sigmoid,
        "activation_prime": sigmoid_prime,
    },
    {
        "sizes": sizes,
        "activation": hyp_tan,
        "activation_prime": hyp_tan_prime,
    },
    {
        "sizes": sizes,
        "activation": softsign,
        "activation_prime": softsign_prime,
    },
]


training_data, test_data, validation_data = load_data(
    file_path="./neural-networks-and-deep-learning/data/mnist.pkl.gz"
)
for net in nets:
    net = Network(**net)
    net.SGD(
        training_data=training_data,
        batch_size=100,
        epochs=10,
        gradient_accumulation_steps=10,
        eta=0.5,
        test_data=test_data,
    )
