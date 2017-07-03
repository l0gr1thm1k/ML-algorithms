import logging
import numpy as np
import time

from collections import defaultdict
from tqdm import tqdm
from mla.utils.main import batch_iterator


"""
Reference
Gradient descent optimization algorithms - http://sebastianruder.com/optimizing-gradient-descent/index.html
"""


class Optimizer(object):

    def optimize(self, network):
        loss_history = []
        for i in range(network.max_epochs):
            if network.shuffle:
                network.shuffle_dataset()

            start_time = time.time()
            loss = self.train_epoch(network)
            loss_history.append(loss)
            if network.verbose:
                msg = "Epoch: %s, train loss: %s" % (i, loss)
                if network.log_metric:
                    msg += ", train %s: %s" % (network.metric_name, network.error())
                msg += ", elapsed: %s sec." % (time.time() - start_time)
                logging.info(msg)
        return loss_history

    def update(self, network):
        """Performs an update on parameters"""
        raise NotImplementedError

    def train_epoch(self, network):
        losses = []

        # create batch iterator
        x_batch = batch_iterator(network.x, network.batch_size)
        y_batch = batch_iterator(network.y, network.batch_size)

        batch = zip(x_batch, y_batch)
        if network.verbose:
            batch = tqdm(batch)

        for x, y in batch:
            loss = np.mean(network.update(x, y))
            self.update(network)
            losses.append(loss)

        epoch_loss = np.mean(losses)
        return epoch_loss

    def train_batch(self, network, x, y):
        loss = np.mean(network.update(x, y))
        self.update(network)
        return loss

    def setup(self, network):
        """
        Creates additional variables
        Must be called before the optimization step
        """
        raise NotImplementedError
