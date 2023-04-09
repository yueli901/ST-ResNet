class MinMaxNormalization(object):
    """
    @fit: get min max from input
    @transform: minmax normalization
        [min, max]--> [-1, 1]
    @fit_transform: sequentially perform @fit and @transform
    @inverse_transform: inverse minmax normalization
        [-1,1] --> [min, max]
    """
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X
