# coding:utf-8

import numpy as np
import tensorflow as tf
import keras.backend as K

class Tensorflow_Model(object):
    
    """
    Implement the tensorflow model of stacking.
    """

    def __init__(self, model_path, output_folder=None, result_type="label"):
        """
        Constructor.
        @ param `model_path` location of the model.
        @ param `output_folder` where to save the stacked model.
        @ param `result_type` "label" or "probs"
        """
        self.model_path = model_path
        self.output_folder = output_folder
        self.result_type = result_type

    def test(self, raw_data):
        model_name = self.model_path.split("/")[-1]
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("%s.meta" % self.model_path)
            saver.restore(sess, self.model_path)
            x = sess.graph.get_tensor_by_name("x:0")
            predict_op = sess.graph.get_tensor_by_name('y_tilda:0')
            result = sess.run(predict_op, feed_dict={x:raw_data, K.learning_phase():0})
            if self.output_folder != None:
                saver.save(sess, "%s./%s.ckpt" % (self.output_folder, model_name))
            # TODO: Using evaluate function, print the accuracy of each model
        
        return np.argmax(result, axis=1) if self.result_type=="label" else result
