import os
import time
import random
import jieba as jb
import numpy as np
import jieba.analyse
import tensorflow as tf
import tensorflow.keras as K
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical

def load_data(path):
    """
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for line in f.readlines():
                sentences.append(line)
                target.append(labels[file[:-4]])

    target = np.array(target)
    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_target = encoder.transform(target)
    dummy_target = to_categorical(encoded_target)

    return sentences, dummy_target


def padding(text_processed, path, max_sequence_length=80):
    """
    数据处理，如果使用 lstm，则可以接收不同长度的序列。
    :text_processed：不定长的 Token 化文本序列，二维list
    :path：数据集路径
    :max_sequence_length：padding 大小，长句截断短句补 0
    :return 处理后的序列，numpy 格式的二维数组
    """
    res = []
    for text in text_processed:
        if len(text) > max_sequence_length:
            text = text[:max_sequence_length]
        else:
            text = text + [0 for i in range(max_sequence_length - len(text))]
        res.append(text)
    return np.array(res)


def processing_data(data_path, validation_split=0.3):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_X,train_y,val_X,val_y
    """
    # --------------- 在这里实现中文文本预处理，包含分词，建立词汇表等步骤 -------------------------------
    # 查看我们创建词汇表的结果
    sentences, target = load_data(data_path)

    # 定义是文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充
    max_sequence_length = 80

    # 使用 jieba 机精确模式分词
    sentences = [".".join(jb.cut(t, cut_all=False)) for t in sentences]
    # 构建词汇表
    vocab_processor = tf.keras.preprocessing.text.Tokenizer(num_words=60000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ',
                                                            oov_token='<UNK>')
    # 要用以训练的文本列表
    vocab_processor.fit_on_texts(sentences)
    # 序列的列表，将 sentences 文本序列化
    text_processed = vocab_processor.texts_to_sequences(sentences)
    vocab_json_string = vocab_processor.to_json()
    # 将词汇表保存路径
    vocab_keras_path = "results/vocab_keras.json"
    file = open(vocab_keras_path, "w")
    file.write(vocab_json_string)
    file.close()

    # 将句子 padding 为固定长度，如果使用lstm则不需要 padding 为固定长度
    text_processed = padding(text_processed, data_path)
    text_target = list(zip(text_processed, target))
    random.shuffle(text_target)
    text_processed[:], target[:] = zip(*text_target)

    # 验证集数目
    val_counts = int(validation_split * len(text_target))

    # 切分验证集
    val_X = text_processed[-val_counts:]
    val_y = target[-val_counts:]
    train_X = text_processed[:-val_counts]
    train_y = target[:-val_counts]

    # --------------------------------------------------------------------------------------------

    return train_X, train_y, val_X, val_y


def dnn_model(train_X, train_y, val_X, val_y, save_model_path):
    """
    创建、训练和保存深度学习模型
    :param train_X: 训练集特征
    :param train_y: 训练集target
    :param test_X: 测试集特征
    :param test_y: 测试集target
    :param save_model_path: 保存模型的路径和名称
    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
    model = K.Sequential()

    # 构建网络层
    # 添加全连接层，输出空间维度64
    model.add(K.layers.Dense(64))

    # 添加激活层，激活函数是 relu
    model.add(K.layers.Activation('relu'))
    model.add(K.layers.Dense(5))
    model.add(K.layers.Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # 训练模型
    history = model.fit(train_X.astype(np.float64), train_y, batch_size=128, epochs=5, validation_data=(val_X, val_y))
    # 保存模型（请写好保存模型的路径及名称）
    model.save(save_model_path)
    plot_training_history(history)

    # --------------------------------------------------------------------------------------------

    return


def plot_training_history(res):
    """
    绘制模型的训练结果
    :param res: 模型的训练结果
    :return:
    """
    # 绘制模型训练过程的损失和平均损失
    # 绘制模型训练过程的损失值曲线，标签是 loss
    plt.plot(res.history['loss'], label='loss')

    # 绘制模型训练过程中的平均损失曲线，标签是 val_loss
    plt.plot(res.history['val_loss'], label='val_loss')

    # 绘制图例,展示出每个数据对应的图像名称和图例的放置位置
    plt.legend(loc='upper right')

    # 展示图片
    plt.show()

    # 绘制模型训练过程中的的准确率和平均准确率
    # 绘制模型训练过程中的准确率曲线，标签是 acc
    plt.plot(res.history['accuracy'], label='accuracy')

    # 绘制模型训练过程中的平均准确率曲线，标签是 val_acc
    plt.plot(res.history['val_accuracy'], label='val_accuracy')

    # 绘制图例,展示出每个数据对应的图像名称，图例的放置位置为默认值。
    plt.legend()

    # 展示图片
    #     plt.show()
    plt.save('results_img/test101.png')
    return


def evaluate_mode(val_X, val_y, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param test_X: 测试集特征
    :param test_y: 测试集target
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    model = K.models.load_model(save_model_path)
    # 获取验证集的 loss 和 accuracy
    loss, accuracy = model.evaluate(val_X, val_y)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    # ---------------------------------------------------------------------------


def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = "./dataset"  # 数据集路径
    save_model_path = "results/model.h5"  # 保存模型路径和名称
    validation_split = 0.2  # 验证集比重

    # 获取数据、并进行预处理
    train_X, train_y, val_X, val_y = processing_data(data_path, validation_split=validation_split)

    # 开始时间
    start = time.time()
    # 数据预处理
    data_path = "./dataset/"
    # 训练模型，获取训练过程和训练后的模型
    history, model = dnn_model(train_X, train_y, val_X, val_y)
    # 打印模型概况和模型训练总数长
    model.summary()
    print("模型训练总时长：", time.time() - start)

    # 评估模型
    evaluate_mode(val_X, val_y, save_model_path)


if __name__ == '__main__':
    main()