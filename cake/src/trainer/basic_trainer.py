import sys
import math
import datetime
import tensorflow as tf
from keras import backend as K


class BasicTrainer(object):
    """
    TODO: need to support keras and tf at the same time.
    """
    optimizer_dict = {"adam": "tf.train.AdamOptimizer"}
    EARLY_STOP_THRESHOLD = 1000

    def __init__(self, **kwargs):

        # Init Session
        self.sess = tf.Session()
        K.set_session(self.sess)

        # Build model
        model_name = kwargs["name"]
        _module = __import__("cake.src.modelzoo.%s"%model_name.lower(), fromlist=["%s"%model_name])
        self.model = getattr(_module, model_name)(**kwargs)

        # Model saver setup
        self.model_save_path = kwargs["model_save_path"]
        self.saver = tf.train.Saver(tf.global_variables())

        # Setup optimizer
        self.loss_layer = self.model.loss_layer
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = self.optimizer.minimize(self.loss_layer)
        self.sess.run(tf.global_variables_initializer())
        self.best_loss = (-1, math.inf) # Initialize loss tracking for earlystop.

    def train_step(self, x_batch, y_batch, step):
        """
        A single training step
        """
        feed_dict = {
          self.model.x: x_batch,
          self.model.y_hat: y_batch,
          K.learning_phase(): 1
        }
        _, loss = self.sess.run([self.train_op, self.loss_layer], feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("%s: step %i, loss %f" % (time_str, step, loss))
        

    def dev_step(self, x_batch, y_batch, step, early_stop=True):
        """
        A single evaluation step
        """
        feed_dict = {
          self.model.x: x_batch,
          self.model.y_hat: y_batch,
          K.learning_phase(): 0
        }
        loss = self.sess.run(self.loss_layer, feed_dict)

        if early_stop:
            return self.is_early_stop(step, loss)
        return False

    def is_early_stop(self, step, loss):
        
        stop_flag = False
        if loss < self.best_loss[1]:
            self.best_loss = (step, loss)
            self.save_sess_2_path()
        elif step - self.best_loss[0] >= BasicTrainer.EARLY_STOP_THRESHOLD:
            print("Best model occurs at step: %i with loss = %.03f" 
                % (self.best_loss[0], self.best_loss[1]))
            stop_flag = True
        return stop_flag

    def save_sess_2_path(self):
        save_prefix = "%smodel" % self.model_save_path
        path = self.saver.save(self.sess, save_prefix, global_step=self.best_loss[0])
        print("Saved winner model checkpoint to %s\n" % path)

    def close_session(self):
        self.sess.close()
