import tensorflow as tf
from tf_dataset import get_data_from_records_3
from cnn_net.mobilenetv1 import Model
import os


def get_losses(predictions, gt_labels, weigh_decay):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels, logits=predictions)
    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
    l2_loss = tf.multiply(tf.add_n(l2_loss_vars), weigh_decay)
    total_loss = tf.add(loss, l2_loss)
    return loss, l2_loss, total_loss


def compute_metrics(predictions, gt_labels):
    pred_max = tf.argmax(predictions, axis=-1)
    correct = tf.equal(pred_max, gt_labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def config_learning_rate():
    global_step = tf.Variable(0, trainable=False)
    # lr = tf.train.exponential_decay(0.0002,
    #                                 global_step,
    #                                 decay_steps=200,
    #                                 decay_rate=0.95,
    #                                 staircase=False)
    boundaries = [5000, 20000, 100000]
    values = [0.001, 0.0002, 0.0001, 0.0001]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
    return global_step, lr


def optimize(loss):
    global_step, lr = config_learning_rate()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step)
    return global_step, train_op, lr


def sess_train():
    label2class_dict = {
        0: "其他垃圾/一次性快餐盒",
        1: "其他垃圾/污损塑料",
        2: "其他垃圾/烟蒂",
        3: "其他垃圾/牙签",
        4: "其他垃圾/破碎花盆及碟碗",
        5: "其他垃圾/竹筷",
        6: "厨余垃圾/剩饭剩菜",
        7: "厨余垃圾/大骨头",
        8: "厨余垃圾/水果果皮",
        9: "厨余垃圾/水果果肉",
        10: "厨余垃圾/茶叶渣",
        11: "厨余垃圾/菜叶菜根",
        12: "厨余垃圾/蛋壳",
        13: "厨余垃圾/鱼骨",
        14: "可回收物/充电宝",
        15: "可回收物/包",
        16: "可回收物/化妆品瓶",
        17: "可回收物/塑料玩具",
        18: "可回收物/塑料碗盆",
        19: "可回收物/塑料衣架",
        20: "可回收物/快递纸袋",
        21: "可回收物/插头电线",
        22: "可回收物/旧衣服",
        23: "可回收物/易拉罐",
        24: "可回收物/枕头",
        25: "可回收物/毛绒玩具",
        26: "可回收物/洗发水瓶",
        27: "可回收物/玻璃杯",
        28: "可回收物/皮鞋",
        29: "可回收物/砧板",
        30: "可回收物/纸板箱",
        31: "可回收物/调料瓶",
        32: "可回收物/酒瓶",
        33: "可回收物/金属食品罐",
        34: "可回收物/锅",
        35: "可回收物/食用油桶",
        36: "可回收物/饮料瓶",
        37: "有害垃圾/干电池",
        38: "有害垃圾/软膏",
        39: "有害垃圾/过期药物"
    }
    record_file = "/home/kun/PycharmProjects/garbage.tfrecord"
    train_epochs = 100
    train_bs = 32
    num_classes = 40
    train_samples = 14802
    train_weight_decay = 0.00001
    input_shape = [224, 224]
    log_dir = "logs"
    model_dir = "model"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    train_summary = set()
    train_images, train_labels, _ = get_data_from_records_3(
        record_file,
        num_samples=train_samples, batch_size=train_bs, out_shape=[224, 224],
        num_classes=num_classes, label2class=label2class_dict,
        is_train=True, aug=True, data_format="channels_last",
        num_epochs=None, num_readers=4, num_threads=8
    )
    train_summary.add(tf.summary.image('image', train_images))
    model = Model(num_classes=num_classes, name="MobileNetV1", data_format="channels_last", reuse=None)
    train_outputs = model.forward(train_images, is_train=True)
    train_loss, train_l2_loss, train_total_loss = get_losses(
        train_outputs, train_labels, train_weight_decay)
    global_step, train_op, lr = optimize(train_total_loss)
    train_summary.add(tf.summary.scalar('train loss', train_loss))
    train_summary.add(tf.summary.scalar('train l2_loss', train_l2_loss))
    train_summary.add(tf.summary.scalar('train total_loss', train_total_loss))
    train_summary.add(tf.summary.scalar('lr', lr))

    train_acc = compute_metrics(train_outputs, train_labels)
    train_summary.add(tf.summary.scalar('train_acc', train_acc))

    saver = tf.train.Saver(tf.trainable_variables())
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                            intra_op_parallelism_threads=0, inter_op_parallelism_threads=0
                            )
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.latest_checkpoint(model_dir)
        if ckpt:
            print('restore from model {}'.format(ckpt))
            saver.restore(sess, ckpt)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        train_summary_op = tf.summary.merge(list(train_summary))
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        for epoch in range(train_epochs):
            print("start epoch {} ....".format(epoch+1))
            for i in range(train_samples // train_bs):
                _, global_step_val, lr_val, train_loss_val, train_l2_loss_val, train_acc_val, train_summary_str \
                    = sess.run([train_op, global_step, lr, train_loss, train_l2_loss, train_acc, train_summary_op])
                if i % 100 == 0:
                    summary_writer.add_summary(train_summary_str, global_step_val)
                    print("step: %ld, lr: %.6f, train loss: %.3f, train_l2_loss: %.3f, train acc: %.3f" % (
                        global_step_val, lr_val, train_loss_val, train_l2_loss_val, train_acc_val))
                    if train_loss_val < 0.02:
                        print("Low loss, stopping now...")
                        saver.save(sess, "{}/model.ckpt-{}".format(log_dir, str(global_step_val)))
                        break
                if i % 1000 == 0:
                    saver.save(sess, "{}/model.ckpt-{}".format(log_dir, str(global_step_val)))
        saver.save(sess, "{}/model.ckpt-{}".format(log_dir, str(global_step_val)))
        coord.request_stop()
        coord.join(threads)


def input_pipeline(record_file, num_samples, batch_size, out_shape, num_classes,
                   label2class, is_train=True, aug=True, data_format="channels_last"):
    def input_fn():
        images, labels, _ = get_data_from_records_3(
            record_file,
            num_samples=num_samples, batch_size=batch_size, out_shape=[224, 224],
            num_classes=num_classes, label2class=label2class,
            is_train=True, aug=True, data_format="channels_last",
            num_epochs=None, num_readers=4, num_threads=8
        )

        return images, labels

    return input_fn


def model_fn(features, labels, mode, params):
    tf.summary.image('image', features)
    model = Model(num_classes=params['num_classes'], name="MobileNetV1",
                  data_format="channels_last", reuse=None)
    outputs = model.forward(features, is_train=(mode==tf.estimator.ModeKeys.TRAIN))
    predicted_class = tf.argmax(outputs, axis=-1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_id": predicted_class[:, tf.newaxis],
            "probabilities": tf.nn.softmax(outputs, axis=-1),
            "logits": outputs
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=outputs)
    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
    l2_loss = tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss')
    total_loss = tf.add(loss, l2_loss, name="total_loss")
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class, name="acc_op")
    metrics = {"accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=total_loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def estimator_train():
    pass


if __name__ == "__main__":
    sess_train()
