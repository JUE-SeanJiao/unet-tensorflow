#coding:utf-8

import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import model
import time

h = 224
w = 224
c_image = 3
c_label = 1
n_class = 5
image_mean = [127, 129, 128]

parser = argparse.ArgumentParser()
parser.add_argument('--trn_dir',
                    default='./data_train.csv')

parser.add_argument('--val_dir',
                    default='./data_val.csv')

parser.add_argument('--model_dir',
                    default='./model')

parser.add_argument('--epochs',
                    type=int,
                    default=500)

parser.add_argument('--epochs_per_eval',
                    type=int,
                    default=1)
parser.add_argument('--logdir',
                    default='./logs')

parser.add_argument('--batch_size',
                    type=int,
                    default=32)

parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3)

parser.add_argument('--decay_rate',
                    type=float,
                    default=0.95)

parser.add_argument('--decay_step',
                    type=int,
                    default=5)

parser.add_argument('--random_seed',
                    type=int,
                    default=1234)

parser.add_argument('--gpu',
                    type=str,
                    default=1)

flags = parser.parse_args()



def set_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def data_augmentation(image, label):
    image_label = tf.concat([image, label], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(image_label)
    maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
    image = maybe_flipped[:, :, :-1]
    label = maybe_flipped[:, :, -1]

    image = tf.image.random_brightness(image, 0.9)
    # image = tf.image.random_hue(image, 0.3)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    return image, label


def read_csv(queue, augmentation=True):
    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = csv_reader.read(queue)

    image_path, label_path = tf.decode_csv(csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    annot_file = tf.read_file(label_path)

    try:
        image = tf.image.decode_jpeg(image_file, channels=3)
    except:
        print('Image: ', image_path)
    image = tf.image.resize_images(image, (h, w))
    image.set_shape([h, w, c_image])
    image = tf.cast(image, tf.float32)

    try:
        annot = tf.image.decode_jpeg(annot_file, channels=1)
    except:
        print('Label: ', label_path)
    annot = tf.image.resize_images(annot, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    annot = tf.cast(annot, tf.float32)
    annot.set_shape([h, w, c_label])

    image_path = tf.string_split([image_path], '/').values[-1]

    if augmentation:
        image, label = data_augmentation(image, annot)
    else:
        pass

    return image, annot, image_path


def loss_CE(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int32)

    y_true = tf.one_hot(y_true[..., 0], n_class)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    # class_weights = tf.constant([0.1, 0.2, 0.2, 0.2, 0.2])
    # weights = tf.gather(class_weights, y_true)
    # cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_true, y_pred, weights)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def loss_IOU(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true[..., 0], n_class)
    pred_flat = tf.reshape(y_pred, [-1, h * w * n_class])
    true_flat = tf.reshape(y_true, [-1, h * w * n_class])
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    union = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) - intersection + 1e-7
    iou = tf.reduce_mean(intersection / union)
    return 1-iou


def visualize_labels(labels):
    labels = tf.cast(labels[..., 0], tf.int32)
    table = tf.constant([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]], tf.int32)
    out = tf.nn.embedding_lookup(table, labels)
    out = tf.cast(out, tf.uint8)
    return out


def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)
    return optimizer.minimize(loss, global_step=global_step)


def main(flags):
    cfg = set_config()
    current_time = time.strftime("%m/%d/%H/%M/%S")
    trn_logdir = os.path.join(flags.logdir, "trn", current_time)
    val_logdir = os.path.join(flags.logdir, "val", current_time)

    trn = pd.read_csv(flags.trn_dir)
    n_trn = trn.shape[0]
    n_trn_step = n_trn // flags.batch_size

    val = pd.read_csv(flags.val_dir)
    n_val = val.shape[0]
    n_val_step = n_val // flags.batch_size

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, h, w, c_image], name='X')
    y = tf.placeholder(tf.float32, shape=[None, h, w, 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    logits = model.unet(X, mode)
    # logits = model.depthwise_unet(X, mode)
    # pred = tf.nn.sigmoid(logits, name='pred')
    pred = tf.nn.softmax(logits, name='pred')

    ce_loss = loss_CE(logits, y)
    iou_loss = loss_IOU(pred, y)
    loss = ce_loss+iou_loss
    tf.summary.scalar("Cross-entropy loss", ce_loss)
    tf.summary.scalar('IOU loss', iou_loss)
    tf.summary.scalar('Total loss', loss)

    #####
    # print(pred.shape)
    pred = tf.cast(tf.argmax(pred, axis=-1), tf.float32)
    pred = tf.expand_dims(pred, axis=-1)
    # print(pred.shape)
    ####

    pred_oh = tf.one_hot(tf.cast(pred, tf.int32), n_class)
    pred_flat = tf.reshape(pred_oh, [-1, h * w * n_class])
    true = tf.one_hot(tf.cast(y[..., 0], tf.int32), n_class)
    true_flat = tf.reshape(true, [-1, h * w * n_class])

    TP = pred_flat * true_flat
    precision = tf.reduce_sum(TP) / (tf.reduce_sum(pred_flat) + 1e-7)
    recall = tf.reduce_sum(TP) / (tf.reduce_sum(true_flat) + 1e-7)
    f1_score = 2 / (1 / precision + 1 / recall + 1e-7)
    tf.summary.scalar("precision", precision)
    tf.summary.scalar("recall", recall)
    tf.summary.scalar("f1_score", f1_score)

    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', mode)
    tf.add_to_collection('outputs', pred)

    tf.summary.image('Input Image', X, max_outputs=8)
    # tf.summary.image('Label', y, max_outputs=8)
    y_view = visualize_labels(y)
    tf.summary.image('Label', y_view, max_outputs=8)
    # tf.summary.image('Predicted Image', pred, max_outputs=8)
    pred_view = visualize_labels(pred)
    tf.summary.image('Prediction', pred_view, max_outputs=8)

    tf.summary.histogram('Predicted Image', pred)

    # global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                               tf.cast(n_trn_step * flags.decay_step, tf.int32),
                                               flags.decay_rate, staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)
    with tf.control_dependencies(update_ops):
        training_op = train_op(loss, learning_rate, global_step)

    summary_op = tf.summary.merge_all()

    # -------------------------------------------- Training -------------------------------------------------

    trn_csv = tf.train.string_input_producer([flags.trn_dir])
    val_csv = tf.train.string_input_producer([flags.val_dir])

    trn_image, trn_label, trn_path = read_csv(trn_csv)
    val_image, val_label, val_path = read_csv(val_csv, augmentation=False)

    X_trn_batch_op, y_trn_batch_op, X_trn_batch_path = tf.train.shuffle_batch([trn_image, trn_label, trn_path],
                                                                              batch_size=flags.batch_size,
                                                                              capacity=flags.batch_size*5,
                                                                              min_after_dequeue=flags.batch_size*2,
                                                                              allow_smaller_final_batch=True)

    X_val_batch_op, y_val_batch_op, X_val_batch_path = tf.train.shuffle_batch([val_image, val_label, val_path],
                                                                              batch_size=flags.batch_size,
                                                                              capacity=flags.batch_size*5,
                                                                              min_after_dequeue=flags.batch_size*2,
                                                                              allow_smaller_final_batch=True)

    with tf.Session(config=cfg) as sess:
        trn_writer = tf.summary.FileWriter(trn_logdir, sess.graph)
        val_writer = tf.summary.FileWriter(val_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        model_dir_ = flags.model_dir
        trained_epoch = 0

        # model_dir_ = 'model_512_200'
        # trained_epoch = int(model_dir_.split('_')[-1])

        if os.path.exists(model_dir_) and tf.train.checkpoint_exists(model_dir_):
            latest_check_point = tf.train.latest_checkpoint(model_dir_)
            saver.restore(sess, latest_check_point)
            print('Load saved model: {}'.format(model_dir_))
            print('Restore training......')
        else:
            print('No model found.')
            print('Start training......')
            try:
                os.rmdir(flags.model_dir)
            except:
                pass
            os.mkdir(flags.model_dir)

        try:
            #global_step = tf.train.get_global_step(sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(trained_epoch, trained_epoch+flags.epochs):
                for step in range(n_trn_step):
                    print('=================================== training ===================================')
                    # print(sess.run([learning_rate])[0])
                    X_trn, y_trn, X_trn_path = sess.run([X_trn_batch_op, y_trn_batch_op, X_trn_batch_path])
                    # print(X_trn.shape)
                    # np.set_printoptions(threshold=np.inf)
                    print(X_trn_path)
                    # print(np.amax(X_trn, axis=(1, 2, 3)))
                    # print(np.amax(y_trn, axis=(1, 2, 3)))

                    _, loss_val, step_summary, pre_val, rec_val = sess.run(
                        [training_op, loss, summary_op, precision, recall], feed_dict={X: X_trn, y: y_trn, mode: True})

                    # print(np.max(y_trn, axis=(1, 2, 3)))
                    # print(np.max(pred_val, axis=(1, 2, 3)))

                    trn_writer.add_summary(step_summary, epoch*n_trn_step + step)
                    print('epoch:{} step:{}/{}, loss: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch+1, step+1, epoch*n_trn_step+step+1, loss_val, pre_val, rec_val))

                for step in range(n_val_step):
                    print('=================================== validation ===================================')
                    X_val, y_val, X_val_path = sess.run([X_val_batch_op, y_val_batch_op, X_val_batch_path])
                    print(X_val_path)
                    loss_val, step_summary, pre_val, rec_val = sess.run([loss, summary_op, precision, recall], feed_dict={X: X_val, y: y_val, mode: False})

                    val_writer.add_summary(step_summary, epoch * n_trn_step + step * n_trn // n_val)
                    print('epoch:{} step:{}, loss: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch + 1, step + 1, loss_val, pre_val, rec_val))

                if (epoch + 1) % 2 == 0:
                    saver.save(sess, '{}/model{:d}.ckpt'.format(flags.model_dir, epoch+1))
                    print('save model {:d}'.format(epoch+1))
                    tf.train.write_graph(sess.graph_def, '', 'disk_graph.pb')

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model_final.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    main(flags)
