import nn


class PerceptronModel(object):
    """
    A binary perceptron for +1/-1 classification.
    """
    def __init__(self, dimensions):
        """
        Initialize with weight parameter of shape (1×dimensions).
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return the current weight parameter.
        """
        return self.w

    def run(self, x):
        """
        Compute scores for input x: shape (batch_size×1).
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Return +1 if score ≥ 0, else -1.
        """
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Perceptron training until convergence on the dataset.
        """
        while True:
            converged = True
            for x, y in dataset.iterate_once(1):
                label = nn.as_scalar(y)
                pred = self.get_prediction(x)
                if pred != label:
                    self.w.update(x, label)
                    converged = False
            if converged:
                break


class RegressionModel(object):
    """
    A 3-layer MLP to approximate sin(x) on [-2π, 2π].
    Architecture: 1 → 100 → 50 → 1 with ReLU activations.
    """
    def __init__(self):
        self.w1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 50)
        self.b2 = nn.Parameter(1, 50)
        self.w3 = nn.Parameter(50, 1)
        self.b3 = nn.Parameter(1, 1)

    def run(self, x):
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        return nn.AddBias(nn.Linear(h2, self.w3), self.b3)

    def get_loss(self, x, y):
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        learning_rate = 0.01
        threshold = 0.02
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        batch_size = 20
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, params)
                for p, g in zip(params, grads):
                    p.update(g, -learning_rate)
            full_x = nn.Constant(dataset.x)
            full_y = nn.Constant(dataset.y)
            if nn.as_scalar(self.get_loss(full_x, full_y)) < threshold:
                return


class DigitClassificationModel(object):
    """
    A 2-hidden-layer MLP for MNIST digit classification.
    Architecture: 784 → 256 → 128 → 10 with ReLU activations.
    """
    def __init__(self):
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        return nn.AddBias(nn.Linear(h2, self.w3), self.b3)

    def get_loss(self, x, y):
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        learning_rate = 0.05
        batch_size = 100
        target_acc = 0.97
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, params)
                for p, g in zip(params, grads):
                    p.update(g, -learning_rate)
            if dataset.get_validation_accuracy() >= target_acc:
                break


class LanguageIDModel(object):
    """
    An RNN model for single-word language identification with an extra dense layer.
    """
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_size = 128
        # RNN parameters
        self.W_initial = nn.Parameter(self.num_chars, self.hidden_size)
        self.b_initial = nn.Parameter(1, self.hidden_size)
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        # Extra dense layer
        self.W_dense = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b_dense = nn.Parameter(1, self.hidden_size)
        # Output layer
        self.W_out = nn.Parameter(self.hidden_size, len(self.languages))
        self.b_out = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        # Initial hidden state
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W_initial), self.b_initial))
        # Recurrent updates
        for x in xs[1:]:
            h = nn.ReLU(
                nn.Add(
                    nn.Linear(x, self.W_initial),
                    nn.Linear(h, self.W_hidden)
                )
            )
        # Extra dense transform
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h, self.W_dense), self.b_dense))
        # Final logits
        return nn.AddBias(nn.Linear(h2, self.W_out), self.b_out)

    def get_loss(self, xs, y):
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        learning_rate = 0.1
        batch_size = 50
        target_acc = 0.81
        params = [
            self.W_initial, self.b_initial,
            self.W_hidden,
            self.W_dense, self.b_dense,
            self.W_out, self.b_out
        ]
        while True:
            for xs, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(xs, y)
                grads = nn.gradients(loss, params)
                for p, g in zip(params, grads):
                    p.update(g, -learning_rate)
            if dataset.get_validation_accuracy() >= target_acc:
                break
