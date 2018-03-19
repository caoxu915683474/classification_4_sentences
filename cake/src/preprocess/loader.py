


class DataLoader(object):
    """
    Interface class for dataloaders from different tasks.
    """
    def __init__(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def load_train(self):
        raise NotImplementedError

    def load_test(self):
        raise NotImplementedError

