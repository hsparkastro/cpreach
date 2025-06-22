import math


# def cosine_annealing(num_epochs):

#     def cosine_annealing(epoch):
#         return 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
#     return cosine_annealing


class ExponentialDecay:
    """ Exponential decay learning rate schedule. """

    def __init__(self, num_epochs, decay_rate=4.0):
        self.num_epochs = num_epochs
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        return math.exp(-self.decay_rate * epoch / self.num_epochs)

    def __str__(self):
        return f"ExponentialDecay(decay_rate={self.decay_rate})"
