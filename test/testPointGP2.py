import sys
sys.path.append("./")
import numpy as np
from geometry.lineGP2 import LineGP2
from geometry.pointGP2 import PointGP2


def test_point_initialization():
    point = PointGP2(array=[4, 5, 1], name='P1')
    assert isinstance(point, PointGP2)
    assert np.array_equal(point.vector, np.array([4, 5, 1]))
    assert point.element_name == 'P1'


def test_point_initialization_no_homogene():
    point = PointGP2(array=[8, 10, 2], name='P1')
    assert isinstance(point, PointGP2)
    assert np.array_equal(point.vector, np.array([4, 5, 1]))
    assert point.element_name == 'P1'


def test_calculating_line():
    pointP1 = PointGP2(array=[1, 2, 1], name='P1')
    pointP2 = PointGP2(array=[8, 10, 2], name='P2')
    result = pointP1.calcularLinea(pointP2, name='L')
    assert isinstance(result, LineGP2)
    assert np.array_equal(result.vector, np.array([-3, 3, -3]))
    assert result.element_name == 'L'

# Puedes agregar más pruebas según lo necesites
