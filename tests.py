import model
import numpy as np


def test_Pooling_01():
    lay = model.Pooling(size=(2, 2), step=1, mode='max')
    arr = np.array([[1, 2, 5, 6],
                    [3, 2, 1, 8],
                    [7, 0, 6, 0]]).reshape((1, 3, 4, 1))
    out = lay(arr)
    expected = np.array([[3, 5, 8],
                         [7, 6, 8]]).reshape((1, 2, 3, 1))
    np.testing.assert_array_equal(out, expected)
