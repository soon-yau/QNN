import tensorflow as tf
import os
from functools import partial

from cifar10 import Cifar10DataSet
import config
from mobilenet import MobileNet
tf.logging.set_verbosity(tf.logging.INFO)
num_training_per_epoch=0

def input_specs(mode, num_epochs):

    def input_fn(mode, batch_size=128):
        with tf.device('/cpu:0'):

            image_batch, label_batch = dataset.make_batch(batch_size)

            return image_batch, label_batch
    print("input function mode:", mode)
    ''' training '''
    if mode=='train':
        dataset=Cifar10DataSet(
            data_dir=config.data_dir, 
            subset=mode, 
            use_distortion=True)

        train_input_fn = partial(input_fn, mode, config.train_batch_size)
        global num_training_per_epoch
        num_training_per_epoch=dataset.num_examples_per_epoch(mode)
        train_steps = num_epochs*num_training_per_epoch/config.train_batch_size
        spec = tf.estimator.TrainSpec(
                        input_fn=train_input_fn, 
                        max_steps=train_steps)
    elif mode=='eval':
        ''' evaluation '''
        dataset=Cifar10DataSet(
            data_dir=config.data_dir, 
            subset=mode, 
            use_distortion=False)

        eval_input_fn = partial(input_fn, mode, config.eval_batch_size)
        eval_steps = dataset.num_examples_per_epoch(mode)/config.eval_batch_size
        print("eval:")
        print("num exampels: ", dataset.num_examples_per_epoch(mode))
        print("batch size:", config.eval_batch_size)
        print("eval_steps:", eval_steps)
        spec = tf.estimator.EvalSpec(
                        input_fn=eval_input_fn, 
                        steps=eval_steps)
    else:
        raise ValueError("Invalid input mode: %s"%mode)

    return spec

def _model_fn(num_bits, features, labels, mode, params):


    is_training = True if mode==tf.estimator.ModeKeys.TRAIN else False
    # create model
    num_classes = 10
    model = MobileNet(num_classes, is_training, num_bits)

    # forward pass
    logits = model.forward_pass(features)

    predictions = {
        'classes':tf.argmax(input=logits, axis=1),
        'probabilities':tf.nn.softmax(logits)}

    # calculate accuracy
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    metrics = {'accuracy':accuracy}

    # loss function
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)     

    # learning rate decay
    global_step = tf.train.get_global_step()
    steps_per_epoch = num_training_per_epoch/config.train_batch_size
    decay_steps = steps_per_epoch*config.decay_per_epoch
    decay_rate = config.decay_rate

    learning_rate = tf.train.exponential_decay(config.learning_rate, 
        global_step, decay_steps, decay_rate)

    # optimize loss
    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # logging
    tf.summary.scalar("accuracy", accuracy[1])
    tf.summary.scalar("learning_rate", learning_rate)

    # printing
    tensors_to_log = {'learning_rate': learning_rate, 
                      'loss': loss, 
                      'accuracy': accuracy[1]}

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)

    train_hooks = [logging_hook]
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)
   


def train(num_bits, num_epochs):
    # create input spec
    train_spec = input_specs('train', num_epochs)
    eval_spec = input_specs('eval', num_epochs)

    # Session config
    
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    if num_bits is None:
        subdir = "numbits_0"
    else:
        subdir = "numbits_%d"%num_bits

    log_folder = os.path.join(config.job_dir, subdir)
    run_config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(), 
        model_dir=log_folder,
        keep_checkpoint_max=3, 
        log_step_count_steps=1000,
        save_summary_steps=1000)

    # Create estimator
    estimator = tf.estimator.Estimator(
        model_fn=partial(_model_fn, num_bits),
        config=run_config)

    # start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':

    # no quantization
    train(None, config.num_epoch)

    for num_bits in range(8,1,-1):
        train(num_bits, config.num_epoch)
