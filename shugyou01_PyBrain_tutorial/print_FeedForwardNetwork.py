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

    in_layer = LinearLayer(2)
    hidden_layer = SigmoidLayer(3)
    out_layer = LinearLayer(1)

    n.addInputModule(in_layer)
    n.addModule(hidden_layer)
    n.addOutputModule(out_layer)

    in_to_hidden = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

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
