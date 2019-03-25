import numpy as np

import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(4, 3),
            l2 = L.Linear(3, 3),
        )

    def __call__(self, x, y):
        fv = self.fwd(x, y)
        loss = F.mean_squared_error(fv, y)
        return loss

    def fwd(self, x, y):
        return F.sigmoid(self.l1(x))

if __name__ == "__main__":
    model = MyChain()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    
    x = Variable(np.array(range(8)).astype(np.float32).reshape(2, 4))
    h = L.Linear(4, 3)
    y = h(x)
    
    model.cleargrads()
    loss = model(x, y)
    loss.backward()
    optimizer.update()
    
