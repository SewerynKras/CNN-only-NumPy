from model import layers
from model.templates import Variable
import numpy as np


def test_Pooling_max_01():
    lay = layers.Pooling(size=(2, 2), step=1, mode='max')
    arr = np.array([[1, 2, 5, 6],
                    [3, 2, 1, 8],
                    [7, 0, 6, 0]]).reshape((1, 3, 4, 1)).astype("float32")
    out = lay(arr)
    expected = np.array([[3, 5, 8],
                         [7, 6, 8]]).reshape((1, 2, 3, 1)).astype("float32")
    np.testing.assert_array_equal(out, expected)

    expected2 = np.array([[0, 0, 5, 0],
                          [3, 0, 0, 16],
                          [7, 0, 6, 0]]).reshape((1, 3, 4, 1)).astype("float32")
    out2, _ = lay.backward(lay(arr))

    np.testing.assert_array_equal(out2, expected2)


def test_Pooling_mean_02():
    lay = layers.Pooling(size=(2, 2), step=1, mode='mean')
    arr = np.array([[1, 2, 5, 6],
                    [3, 2, 1, 8],
                    [7, 0, 6, 0]]).reshape((1, 3, 4, 1)).astype('float32')
    expected = np.array([[2.0, 2.5, 5.0],
                         [3.0, 2.25, 3.75]]).reshape((1, 2, 3, 1))
    out = lay(arr)

    np.testing.assert_array_equal(out, expected)

    expected2 = np.array([[0.5, 1.125, 1.875, 1.25],
                          [1.25, 2.4375, 3.375, 2.1875],
                          [0.75, 1.3125, 1.5, 0.9375]]).reshape((1, 3, 4, 1)).astype("float32")
    out2, _ = lay.backward(lay(arr))

    np.testing.assert_array_equal(out2, expected2)


def test_Conv_01():
    conv = layers.Conv2D((2, 2), 1, 1)
    inp = np.array([[2, 4, 6],
                    [8, 10, 12],
                    [14, 16, 18]]).reshape((1, 3, 3, 1)).astype("float32")
    filters = np.array([[-1, 0.5],
                        [1, 2]]).reshape((2, 2, 1, 1)).astype('float32')
    biases = np.ones((1,)).astype("float32")
    conv.initialized = True
    conv.weights = Variable(filters)
    conv.biases = Variable(biases)

    expected = np.array([[29, 34],
                         [44, 49]]).reshape((1, 2, 2, 1)).astype("float32")

    out = conv(inp)

    np.testing.assert_array_equal(expected, out)
