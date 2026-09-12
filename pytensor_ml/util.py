import numpy as np

from pytensor import config


class DataLoader:
    """
    Draw shuffled, fixed-size batches from a dataset, cycling indefinitely.

    Calling the loader returns the next ``(X_batch, y_batch)``. Batches are always ``batch_size`` rows: a
    pass that runs off the end of the data is topped up from a freshly shuffled order rather than coming up
    short, so a batch may straddle two epochs.

    Parameters
    ----------
    X : ndarray
        Features, indexed along the first axis.
    y : ndarray
        Targets, indexed along the first axis and aligned row-wise with ``X``.
    batch_size : int
        Rows per batch. Default 64.
    dtype : str, optional
        Dtype to cast ``X`` and ``y`` to. Defaults to ``floatX``.
    random_state : int or numpy Generator, optional
        Seed for the shuffling generator, for reproducible batch sequences.

    Examples
    --------
    Call the loader once per training step; it reshuffles and wraps around on its own, so the loop
    never has to track epoch boundaries:

    .. code-block:: python

        import numpy as np

        from pytensor_ml.util import DataLoader

        X = np.random.default_rng(0).normal(size=(500, 8))
        y = np.random.default_rng(1).normal(size=(500, 1))

        loader = DataLoader(X, y, batch_size=32, random_state=0)
        X_batch, y_batch = loader()
    """

    def __init__(self, X, y, batch_size=64, dtype=None, random_state=None):
        if dtype is None:
            dtype = config.floatX
        self.rng = np.random.default_rng(random_state)

        self.X = X.astype(dtype)
        self.y = y.astype(dtype)
        self.n = X.shape[0]

        self.batch_size = batch_size
        self.indices = np.arange(self.n, dtype="int32")

        # Declared here so the constructor shows the object's full state; reset() assigns them.
        self.cursor: int
        self.epoch: int
        self.reset()

    def shuffle(self):
        self.rng.shuffle(self.indices)

    def move_cursor(self):
        start = self.cursor
        n_wraps, stop = divmod(start + self.batch_size, self.n)

        batch_slice = slice(start, None if n_wraps else stop)
        # Copy, not a view: the refill below reshuffles self.indices in place and would rewrite these rows.
        batch_indices = self.indices[batch_slice].copy()

        # One refilled pass per wrap, since a single pass supplies at most n rows and a batch wider than
        # the dataset crosses several.
        for _ in range(n_wraps):
            self.shuffle()
            self.epoch += 1
            shortfall = self.batch_size - batch_indices.shape[0]
            batch_indices = np.r_[batch_indices, self.indices[:shortfall]]

        self.cursor = (start + self.batch_size) % self.n

        return batch_indices

    def reset(self):
        self.cursor = 0
        self.epoch = 0
        self.shuffle()

    def __call__(self):
        batch_indices = self.move_cursor()
        return self.X[batch_indices], self.y[batch_indices]
