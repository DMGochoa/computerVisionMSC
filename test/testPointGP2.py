import sys
sys.path.append("./")
from geometry.pointGP2 import PointGP2
from geometry.lineGP2 import LineGP2
import numpy as np

def test_point_initialization():
    point = PointGP2(array=[4, 5, 1], nombre='P1')
    assert isinstance(point, PointGP2)
    assert np.array_equal(point.vector, [[4], [5], [1]])
    assert point.nombre_elemento == 'P1'

def test_calculating_line():
    pointP1 = LineGP2(array=[1, 2, 3], nombre='P1')
    pointP2 = LineGP2(array=[4, 5, 6], nombre='P2')
    result = pointP1.calcularLinea(pointP2, nombre='L')
    assert isinstance(result, LineGP2)
    assert np.array_equal(result, [[-3], [6], [-3]])

# Puedes agregar más pruebas según lo necesites
