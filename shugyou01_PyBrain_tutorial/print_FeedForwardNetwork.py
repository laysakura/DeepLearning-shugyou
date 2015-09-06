"""http://pybrain.org/docs/tutorial/netmodcon.html
"""


# standard modules

# 3rd party modules
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

# original modules


def main():
    n = FeedForwardNetwork()

    inLayer = LinearLayer(2)
    hiddenLayer = SigmoidLayer(3)
    outLayer = LinearLayer(1)

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)

    n.sortModules()

    print(">>> print n")
    print(n)

    print(">>> n.activate([1, 2])")
    print(n.activate([1, 2]))

    print(">>> in_to_hidden.params")
    print(in_to_hidden.params)

    print(">>> hidden_to_out.params")
    print(hidden_to_out.params)

    print(">>> n.params")
    print(n.params)


if __name__ == '__main__':
    main()
