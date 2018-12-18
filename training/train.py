import tensorflow as tf
import os
from functools import partial

import cifar10 
import config
from mobilenet import MobileNet
tf.logging.set_verbosity(tf.logging.INFO)
num_training_per_epoch=0


def input_fn(mode, batch_size=128):

    dataset=cifar10.Cifar10DataSet(
        data_dir=config.data_dir, 
        subset=mode, 
        use_distortion=mode=='train')

    with tf.device('/cpu:0'):

        image_batch, label_batch = dataset.make_batch(batch_size)

        return image_batch, label_batch

def input_specs(mode, num_epochs):
    dataset=cifar10.Cifar10DataSet
    ''' training '''
    if mode=='train':

        train_input_fn = partial(input_fn, mode, config.train_batch_size)
        global num_training_per_epoch
        num_training_per_epoch=dataset.num_examples_per_epoch(mode)
        train_steps = num_epochs*num_training_per_epoch/config.train_batch_size
        spec = tf.estimator.TrainSpec(
                        input_fn=train_input_fn, 
                        max_steps=train_steps)
    elif mode=='eval':
        ''' evaluation '''
        batch_size = config.eval_batch_size
        eval_input_fn = partial(input_fn, mode, batch_size)
        eval_steps = dataset.num_examples_per_epoch(mode)/batch_size
        spec = tf.estimator.EvalSpec(
                        input_fn=eval_input_fn, 
                        steps=eval_steps,
                        throttle_secs=100)
    else:
        raise ValueError("Invalid input mode: %s"%mode)

    return spec

def _model_fn(num_bits, features, labels, mode, params):


    is_training =  (mode==tf.estimator.ModeKeys.TRAIN)

    # weight reguralization
    regularizer = tf.contrib.layers.l2_regularizer(scale=config.weight_decay)
    # create model
    num_classes = 10
    model = MobileNet(num_classes, 
                      is_training, num_bits, 
                      width_multiplier=config.width_multiplier, 
                      quant_mode=config.quant_method,
                      conv2d_regularizer=regularizer)

    # forward pass
    logits = model.forward_pass(features)
    predict_class = tf.argmax(input=logits, axis=1)
    #predict_class = tf.Print(predict_class, [predict_class])
    predictions = {
        'classes':predict_class,
        'probabilities':tf.nn.softmax(logits)}

    # calculate accuracy
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    metrics = {'accuracy':accuracy}

    # loss function
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)     


    # reguralization loss
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term

    if mode==tf.estimator.ModeKeys.TRAIN:

        # add fake_quant to 'normal' graph
        if config.quant_method=='tensorflow':
            print("TF quantize create training graph")
            g = tf.get_default_graph()
            tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)

        # learning rate decay
        global_step = tf.train.get_global_step()
        steps_per_epoch = num_training_per_epoch/config.train_batch_size
        decay_steps = steps_per_epoch*config.decay_per_epoch
        decay_rate = config.decay_rate

        learning_rate = tf.train.exponential_decay(config.learning_rate, 
            global_step, decay_steps, decay_rate)

        learning_rate = tf.maximum(learning_rate, config.learning_rate*0.01)
        # optimize loss
        optimizer = tf.train.AdamOptimizer(learning_rate)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, 
                                          global_step=tf.train.get_global_step())


        # logging
        tf.summary.scalar("accuracy", accuracy[1])
        tf.summary.scalar("learning_rate", learning_rate)
        
        # printing
        tensors_to_log = {'learning_rate': learning_rate, 
                          'loss': loss, 
                          'accuracy': accuracy[1]}

        train_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=[train_hook],
            eval_metric_ops=metrics)

    elif mode==tf.estimator.ModeKeys.EVAL:
        if config.quant_method=='tensorflow':
            g = tf.get_default_graph()
            tf.contrib.quantize.create_eval_graph(input_graph=g)

        tf.summary.scalar("accuracy", accuracy[1])
        eval_tensors_to_log = {'eval_loss': loss, 
                          'eval_accuracy': accuracy[1]}
        evaluation_hook = tf.train.LoggingTensorHook(
            tensors=eval_tensors_to_log, every_n_iter=1000)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            evaluation_hooks=[evaluation_hook],
            eval_metric_ops=metrics)
   


def train(num_bits, num_epochs):
    # create input spec
    train_spec = input_specs('train', num_epochs)
    eval_spec = input_specs('eval', num_epochs)

    subdir=config.quant_method

    if num_bits is None:
        subdir+= "/numbits_9"
    else:
        subdir+= "/numbits_%d"%num_bits

    log_folder = os.path.join(config.job_dir, subdir)
    run_config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(), 
        model_dir=log_folder,
        keep_checkpoint_max=5,
        save_checkpoints_steps=1000, 
        log_step_count_steps=1000,
        save_summary_steps=1000)

    # Create estimator
    estimator = tf.estimator.Estimator(
        model_fn=partial(_model_fn, num_bits),
        config=run_config)

    # start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #estimator.evaluate(input_fn=partial(input_fn,'eval', 100), steps=None)

if __name__ == '__main__':

    # no quantization
    train(None, config.num_epoch)

    for num_bits in range(8,1,-1):
        train(num_bits, config.num_epoch)
