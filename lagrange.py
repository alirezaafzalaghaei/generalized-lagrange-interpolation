import numpy as np
from functools import partial

class GeneralizedLagrange:

    def __init__(self, x, y, phi):
        self.x = x
        self.y = y
        self.phi = phi
        
        self.u = lambda t, i, j: (self.phi(t) - self.phi(self.x[j])) / (
            self.phi(self.x[i]) - self.phi(self.x[j]))
        self.prod = lambda t, i: np.prod(
            [self.u(t, i, j) for j in range(len(self.x)) if i != j])
        self.predict = np.vectorize(self.predict_scaler)

    def predict_scaler(self, t):
        return sum(map(lambda i: self.y[i] * self[i](t), range(len(self.x))))

    def __getitem__(self, i):
        return partial(self.prod, i=i)

    def __call__(self, x):
        return self.predict(x)
        
class Lagrange(GeneralizedLagrange):
    def __init__(self, x, y):
        phi = lambda x: x
        super().__init__(x, y, phi)
        
def plot(*, x_sample, y_sample, x_pred, y_pred, y_real):
    import matplotlib.pyplot as plt
    plt.scatter(x_sample, y_sample, label='data')
    plt.plot(x_pred, y_pred, 'r--', label='Lagrange')
    plt.plot(x_pred, y_real, linestyle='-.', label='exact')
    plt.legend()
    print('Mean Absolute Error: {:0.3f}'.format(np.abs(y_pred - y_real).mean()))