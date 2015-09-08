"""XOR論理素子の出力を学習。
"""


# standard modules

# 3rd party modules
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pylab as pl

# original modules


def main():
    N_IN_UNITS = 2
    N_OUT_UNITS = 1
    N_HIDDEN_UNITS = 2

    N_EPOCHS = 1000

    # 分類器の構築。buildNetwork()を使うと、入力層はLinearLayerになる。
    net = buildNetwork(
        N_IN_UNITS, N_HIDDEN_UNITS, N_OUT_UNITS,
        hiddenclass=SigmoidLayer, outclass=SigmoidLayer,
        recurrent=True, bias=True
    )
    net.randomize()

    print(">>> print net")
    print(net)

    # 入出力データ。自明に同じデータ点を重複して持って意味あるのか・・・?
    ds = SequentialDataSet(N_IN_UNITS, N_OUT_UNITS)
    for _ in range(10):
        ds.newSequence()
        ds.appendLinked([0, 0], [0])
        ds.appendLinked([0, 1], [1])
        ds.appendLinked([1, 0], [1])
        ds.appendLinked([1, 1], [0])

    # 学習曲線のインタラクティブ描画準備。
    pl.figure()
    pl.ion()
    pl.hold(True)
    pl.show()

    # 10タイムステップ分の勾配降下。学習曲線も描画しつつ。
    trainer = BackpropTrainer(net, ds)
    for i in range(N_EPOCHS):
        train_err = trainer.train()
        pl.plot(i, train_err, 'o')
        pl.draw()  # 描画自体の負荷がかなり重いので、コメントアウトすると速くなる。

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
