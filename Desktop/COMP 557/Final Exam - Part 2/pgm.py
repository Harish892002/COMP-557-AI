# pgm.py

import numpy as np
import utils


class PGM:
    """
    Part 2 PGM class for COMP/ELEC 440/557.

    Q1: simple linear least‐squares predictor for Y|W,X.
    Q2/Q3: empirical Bayes inference on the true (w,x,y) triples.
    """

    def __init__(self):
        # raw training data
        self.w_train = None
        self.x_train = None
        self.y_train = None

    def train(self, data_w, data_x, data_y):
        """
        Store training triples.  We don't fit a global regressor here—
        for Q1 we’ll use 3-NN, for Q2/Q3 we’ll do empirical sampling.
        """
        self.w_train = data_w
        self.x_train = data_x
        self.y_train = data_y

    def predict_y(self, w, x):
        """
        Q1: 3-nearest-neighbor regressor on (w,x).
        Return the mean y of the three closest training points.
        """
        # compute squared Euclidean distances to all training points
        d2 = (self.w_train - w) ** 2 + (self.x_train - x) ** 2

        # find indices of the 3 smallest distances
        k = 3
        idxs = np.argpartition(d2, k)[:k]

        # predict as the average of their y’s
        return float(self.y_train[idxs].mean())

    def infer_w_given_z(self, z):
        """
        Q2: empirical estimate of P(W | Z=z):
          1) simulate one Z_i = generate_z_given_y(y_i) for each training y_i
          2) select all w_i whose Z_i == z
          3) return their sample mean and std
        """
        # simulate Z for each true y
        zs = np.array([
            utils.generate_z_given_y(float(y_i))
            for y_i in self.y_train
        ])
        w_post = self.w_train[zs == z]
        return float(w_post.mean()), float(w_post.std())

    def infer_w_given_x_z(self, x, z):
        """
        Q3: empirical estimate of P(W | X≈x, Z=z):
          1) pick the K nearest neighbors in training x to the query x
          2) simulate Z for their y's
          3) select those w's with Z == z
          4) return sample mean and std
        """
        K = 5000
        # find indices of the K closest training x's
        idxs = np.argsort(np.abs(self.x_train - x))[:K]
        wK = self.w_train[idxs]
        yK = self.y_train[idxs]

        # simulate Z for those y's
        zK = np.array([
            utils.generate_z_given_y(float(y_i))
            for y_i in yK
        ])
        w_post = wK[zK == z]
        return float(w_post.mean()), float(w_post.std())