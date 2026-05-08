import numpy as np

# Schedulers


# EXponential learning rate scheduler
class ExponentialLR:
    def __init__(self, final_lr, decay_rate, initial_lr=1e-3):
        self.final_lr = final_lr
        self.decay_rate = decay_rate
        self.initial_lr = initial_lr

    def get_lr(self, k):
        return self.final_lr + (self.initial_lr - self.final_lr) * np.exp(
            -self.decay_rate * k
        )
