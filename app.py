from flask import Flask, jsonify
from lda_data_preprocess.clusering import Clustering_algorithm
from lda_data_preprocess.toolfunction import dataset
from lstm_data_preprocess.prepare_data import *
import time
from lstm_data_preprocess.model_helper import *
from lstm_data_preprocess.main import ABLSTM
app = Flask(__name__)


@app.route('/')
def hello_world():

    return 'Hello World!'

@app.route('/recommendation')
def Lda_result():
    a = Clustering_algorithm()
    d = dataset()
    data_samples, data_labels = d.tutorial_dataset('fetch_20newsgroup', 'train', 0.5)
    terms_frequency, topics, keywords = a.lda_data(data_samples=data_samples, n_samples=2000, n_features=1000, n_components=10,
                                            n_top_words=7)

    key_words = jsonify(keywords)
    print(key_words)
    return jsonify(keywords)


@app.route('/')
def recommendation_result():
    # load data
    #dbpedia = tf.data.Dataset('dbpedia')
    x_train, y_train = load_data("./lstm/dataset/dbpedia_data/dbpedia_csv/train.csv", sample_ratio=1e-2, one_hot=False)
    x_test, y_test = load_data("./lstm/dataset/dbpedia_data/dbpedia_csv/test.csv", one_hot=False)

    # data preprocessing
    x_train, x_test, vocab_size, train_words, test_words = data_preprocessing_v2(x_train, x_test, max_len=32,
                                                                                     max_words=50000)
    print("train size: ", len(x_train))
    print("vocab size: ", vocab_size)

    # split dataset to test and dev
    x_test, x_dev, y_test, y_dev, dev_size, test_size = split_dataset(x_test, y_test, 0.1)
    print("Validation Size: ", dev_size)

    config = {
        "max_len": 32,
        "hidden_size": 64,
        "vocab_size": vocab_size,
        "embedding_size": 128,
        "n_class": 15,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "train_epoch": 20
    }

    classifier = ABLSTM(config)
    classifier.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    dev_batch = (x_dev, y_dev)
    start = time.time()

    # view_data = data_Visualization()
    validation_accuracy = list()

    for e in range(config["train_epoch"]):
        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
            # plot the attention weight
            # print(np.reshape(attn, (config["batch_size"], config["max_len"])))
        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc = run_eval_step(classifier, sess, dev_batch)
        print("validation accuracy: %.3f " % dev_acc)
        validation_accuracy.append(dev_acc)

    index = list(np.arange(1, len(validation_accuracy) + 1, 1))
    # view_data.accuracy(index, validation_accuracy)
    print("Training finished, time consumed : ", time.time() - start, " s")
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step(classifier, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))
    return jsonify(test_acc)




if __name__ == '__main__':
    app.run()
