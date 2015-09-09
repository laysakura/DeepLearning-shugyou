"""XOR論理素子の出力を、様々なニューラルネットで学習。

学習速度と精度のを向上させるため、様々なニューラルネットを試す。
データ点の個数とepoch数は固定。

訓練データもテストデータも4通りしかないので、過学習しやすいニューラルネットが有利な点に注意。
"""


# standard modules

# 3rd party modules
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer, TanhLayer, ReluLayer
from pybrain.datasets.sequential import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pylab as pl
from benchmarker import Benchmarker

# original modules


N_IN_UNITS = 2
N_OUT_UNITS = 1


def net1():
    """ベースライン。

    Wの初期値が悪いと、epoch数重ねないと収束しない印象。

    - 隠れ層1
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層2
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    """
    n_hidden1_units = 2
    n_hidden2_units = 2

    # 分類器の構築。buildNetwork()を使うと、入力層はLinearLayerになる。
    net = buildNetwork(
        N_IN_UNITS, n_hidden1_units, n_hidden2_units, N_OUT_UNITS,
        hiddenclass=TanhLayer, outclass=SigmoidLayer,
        bias=True
    )
    net.randomize()

    return net


def net2():
    """ユニット数を増やす。

    ベースラインに比べて遅くないし、収束は速い。

    - 隠れ層1
        - ユニット数: 10
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層2
        - ユニット数: 10
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    """
    n_hidden1_units = 10
    n_hidden2_units = 10

    # 分類器の構築。buildNetwork()を使うと、入力層はLinearLayerになる。
    net = buildNetwork(
        N_IN_UNITS, n_hidden1_units, n_hidden2_units, N_OUT_UNITS,
        hiddenclass=TanhLayer, outclass=SigmoidLayer,
        bias=True
    )
    net.randomize()

    return net


def net3():
    """隠れ層を増やす。

    epoch辺りの処理が遅くなったし、精度も全然上がってない。

    - 隠れ層1
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層2
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層3
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層4
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層5
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    - 隠れ層6
        - ユニット数: 2
        - 活性化関数: tanh
        - バイアス: (よくわかってない)
    """
    n_hidden1_units = 2
    n_hidden2_units = 2
    n_hidden3_units = 2
    n_hidden4_units = 2
    n_hidden5_units = 2
    n_hidden6_units = 2

    # 分類器の構築。buildNetwork()を使うと、入力層はLinearLayerになる。
    net = buildNetwork(
        N_IN_UNITS,
        n_hidden1_units, n_hidden2_units, n_hidden3_units,
        n_hidden4_units, n_hidden5_units, n_hidden6_units,
        N_OUT_UNITS,
        hiddenclass=TanhLayer, outclass=SigmoidLayer,
        bias=True
    )
    net.randomize()

    return net


def net4():
    """net2よりも学習速度を速くするため、正規化線形関数を用いる。

    net2と精度は同じくらい。draw()をコメントアウトすると、わずかに速くなったのも確認できる。

    - 隠れ層1
        - ユニット数: 10
        - 活性化関数: ReLU
        - バイアス: (よくわかってない)
    - 隠れ層2
        - ユニット数: 10
        - 活性化関数: ReLU
        - バイアス: (よくわかってない)
    """
    n_hidden1_units = 10
    n_hidden2_units = 10

    # 分類器の構築。buildNetwork()を使うと、入力層はLinearLayerになる。
    net = buildNetwork(
        N_IN_UNITS, n_hidden1_units, n_hidden2_units, N_OUT_UNITS,
        hiddenclass=ReluLayer, outclass=SigmoidLayer,
        bias=True
    )
    net.randomize()

    return net


def main():
    n_epochs = 50

    # 試したいニューラルネットをコメントイン。
    # net = net1()
    # net = net2()
    # net = net3()
    net = net4()
    print(">>> print net")
    print(net)

    # 入出力データ。
    ds = SupervisedDataSet(N_IN_UNITS, N_OUT_UNITS)
    for _ in range(100):
        ds.appendLinked([0, 0], [0])
        ds.appendLinked([0, 1], [1])
        ds.appendLinked([1, 0], [1])
        ds.appendLinked([1, 1], [0])

    # 学習曲線のインタラクティブ描画準備。
    pl.figure()
    pl.ion()
    pl.hold(True)
    pl.show()

    # N_EPOCHSタイムステップ分の勾配降下。学習曲線も描画しつつ。
    trainer = BackpropTrainer(net, ds)
    with Benchmarker() as bench:
        @bench("net\n")
        def _(_):
            for i in range(n_epochs):
                train_err = trainer.train()
                pl.plot(i, train_err, 'o')
                pl.draw()
                print("Iteration %04d : train_err=%f" % (i, train_err))

    # テストセットで学習結果を確認。
    print([0, 0], net.activate([0, 0]))
    print([0, 1], net.activate([0, 1]))
    print([1, 0], net.activate([1, 0]))
    print([1, 1], net.activate([1, 1]))

    # 学習曲線を出したままプログラムを止める。
    pl.ioff()
    pl.show()


if __name__ == '__main__':
    main()
