"""XOR論理素子の出力を学習。
"""


# standard modules

# 3rd party modules
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import ReluLayer, SigmoidLayer
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer

# original modules


def main():
    N_IN_UNITS =  2
    N_OUT_UNITS = 1
    N_HIDDEN_UNITS = 10

    # 分類器の構築。buildNetwork()を使うと、入力層はLinearLayerになる。
    net = buildNetwork(
        N_IN_UNITS, N_HIDDEN_UNITS, N_OUT_UNITS,
        hiddenclass=ReluLayer, outclass=SigmoidLayer,
        recurrent=True)
    net.randomize()

    print(">>> print net")
    print(net)

    # 入出力データ。自明に同じデータ点を重複して持って意味あるのか・・・?
    ds = SequentialDataSet(N_IN_UNITS, N_OUT_UNITS)
    for _ in range(100):
        ds.addSample([0, 0], [0])
        ds.addSample([0, 1], [1])
        ds.addSample([1, 0], [1])
        ds.addSample([1, 1], [0])

    # 10タイムステップ分の勾配降下。
    trainer = BackpropTrainer(net, ds)
    for _ in range(10):
        print(trainer.train())

    # テストセットで学習結果を確認。
    print([0, 0], net.activate([0, 0]))
    print([0, 1], net.activate([0, 1]))
    print([1, 0], net.activate([1, 0]))
    print([1, 1], net.activate([1, 1]))


if __name__ == '__main__':
    main()
