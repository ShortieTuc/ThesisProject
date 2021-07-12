import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal



'''Class for Gaussian Kernel Regression'''


class GKR:

    def __init__(self, x, y, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b = b

    '''Implement the Gaussian Kernel'''

    def gaussian_kernel(self, z):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

    '''Calculate weights and return prediction'''

    def predict(self, h):
        kernels = np.array([self.gaussian_kernel((np.linalg.norm(xi - h)) / self.b) for xi in self.x])
        weights = np.array([len(self.x) * (kernel / np.sum(kernels)) for kernel in kernels])
        return np.dot(weights.T, self.y) / len(self.x)

    # def visualize_kernels(self):
    #     zsum = np.zeros((120, 120))
    #     plt.figure(figsize=(10, 5))
    #     ax = plt.axes(projection='3d')
    #     for xi in self.x:
    #         x, y = np.mgrid[0:120:120j, 0:120:120j]
    #         xy = np.column_stack([x.flat, y.flat])
    #         z = multivariate_normal.pdf(xy, mean=xi, cov=self.b)
    #         z = z.reshape(x.shape)
    #         zsum += z
    #
    #     ax.plot_surface(x, y, zsum)
    #
    #     ax.set_ylabel('y')
    #     ax.set_xlabel('x')
    #     ax.set_zlabel('Kernel Weights wi')

