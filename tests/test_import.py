"""Test that gb_net_lib can be imported and basic functionality works."""

import gb_net_lib
import numpy as np


def test_import():
    """Test that the module imports successfully."""
    assert hasattr(gb_net_lib, 'bp')
    assert hasattr(gb_net_lib, 'net')
    print("Import successful: bp and net classes found")


def test_bp_basic():
    """Test basic bp functionality."""
    a = gb_net_lib.bp([[1.0, 2.0], [3.0, 4.0]])
    b = gb_net_lib.bp([[0.5], [1.5]])
    
    # Test matrix multiplication
    c = a * b
    assert c.data.shape == (2, 1)
    
    # Test addition
    d = gb_net_lib.bp([[1.0], [2.0]])
    e = gb_net_lib.bp([[3.0], [4.0]])
    f = d + e
    assert np.allclose(f.data, [[4.0], [6.0]])
    
    # Test subtraction
    g = d - e
    assert np.allclose(g.data, [[-2.0], [-2.0]])
    
    # Test squaring
    h = d.sq()
    assert np.allclose(h.data, [[1.0], [4.0]])
    
    # Test sigmoid
    i = gb_net_lib.bp([[0.0]])
    j = i.sg()
    assert np.allclose(j.data, [[0.5]])
    
    # Test relu
    k = gb_net_lib.bp([[-1.0], [2.0]])
    l = k.relu()
    assert np.allclose(l.data, [[0.0], [2.0]])
    
    print("Basic bp operations work")


def test_backprop():
    """Test backpropagation."""
    a = gb_net_lib.bp([[2.0]])
    b = gb_net_lib.bp([[3.0]])
    c = a * b  # 6.0
    loss = c.sq()  # 36.0
    loss.backprop()
    
    # Gradient of x^2 is 2x, so at x=6, grad = 12
    # Then chain rule: d(loss)/da = d(loss)/dc * dc/da = 12 * b = 12 * 3 = 36
    # d(loss)/db = 12 * a = 12 * 2 = 24
    assert np.allclose(a.grad, [[36.0]])
    assert np.allclose(b.grad, [[24.0]])
    print("Backpropagation works")


def test_net_basic():
    """Test basic net functionality."""
    model = gb_net_lib.net(N=2, layers=[4, 1], act='sg')
    struc = model.layer_init()
    
    assert len(struc) == 2
    assert struc[0].data.shape == (4, 2)
    assert struc[1].data.shape == (1, 4)
    print("Net layer initialization works")


def test_net_train():
    """Test net training on simple data."""
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0], [1.0], [1.0], [1.0]]
    
    model = gb_net_lib.net(N=2, layers=[4, 1], act='sg')
    trained_weights = model.train(X, Y, epochs=10, lr=0.5)
    
    assert len(trained_weights) == 2
    print("Net training works")


if __name__ == "__main__":
    test_import()
    test_bp_basic()
    test_backprop()
    test_net_basic()
    test_net_train()
    print("\nAll tests passed!")