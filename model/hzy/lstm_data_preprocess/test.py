import tensorflow as tf
from main import load_data
from prepare_data import data_preprocessing_v2, fill_feed_dict
import pickle
from keras_preprocessing.sequence import pad_sequences
import numpy as np


def make_test_feed_dict0(x, label, keep_prob, batch):
    feed_dict = {x: batch[0],
                 label: batch[1],
                 keep_prob: 1.0}
    return feed_dict


def run_eval_step0(x, label, keep_prob, prediction, sess, batch):
    feed_dict = make_test_feed_dict0(x, label, keep_prob, batch)
    prediction = sess.run(prediction, feed_dict=feed_dict)
    acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return acc


config = {
    "max_len": 32,
    "hidden_size": 64,
    "vocab_size": 50002,
    "embedding_size": 128,
    "n_class": 15,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "train_epoch": 20
}

max_len = 32
x_train, y_train = load_data("./dataset/dbpedia_data/dbpedia_csv/train.csv", sample_ratio=1e-2, one_hot=False)
x_test0, y_test = load_data("./dataset/dbpedia_data/dbpedia_csv/test.csv", one_hot=False)
x_train, x_test, vocab_size, train_words, test_words, tokenizer = data_preprocessing_v2(x_train, x_test0, max_len=32,
                                                                                        max_words=50000)
with open('./dataset/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./dataset/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_idx = tokenizer.texts_to_sequences(x_test0)
test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
print(test_padded)
print(x_test)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model_saved/model.ckpt.meta')
    saver.restore(sess, "./model_saved/model.ckpt")
    graph = tf.get_default_graph()
    name = graph.get_operations()
    print(name)
    x = graph.get_operation_by_name('x')
    label = graph.get_operation_by_name('label')
    keep_prob = graph.get_operation_by_name('keep_prob')
    prediction = graph.get_operation_by_name('prediction')

    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step0(x, label, keep_prob, prediction, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1
    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
