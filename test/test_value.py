from kiwigrad import Value
import torch
import unittest

class TestValue(unittest.TestCase):

    def test_activation_functions(self):
        #kiwigrad
        a = Value(2.)
        b = a.sigmoid()
        c = b.relu()
        d = c.tanh()
        f = d.exp()
        f.backward()
        #pytorch
        at = torch.tensor([2.], requires_grad=True)
        bt = at.sigmoid()
        ct = bt.relu()
        dt = ct.tanh()
        ft = dt.exp()
        ft.backward()
        at_g = float(at.grad)
        #test
        self.assertAlmostEqual(a.grad, at_g)

    def test_more_ops(self):
        #kiwigrad
        a = Value(3.)
        b = Value(3.)
        c = (a * Value(2))
        d = (c - b) ** 2
        e = d.log()
        f = e.exp()
        f.backward()
        #torch
        at = torch.tensor([3.], requires_grad=True)
        bt = torch.tensor([3.], requires_grad=True)
        ct = at * 2
        dt = (ct - bt) ** 2
        et = dt.log()
        ft = et.exp()
        ft.backward()
        at_g = float(at.grad)
        bt_g = float(bt.grad)
        self.assertAlmostEqual(a.grad, at_g)
        self.assertAlmostEqual(b.grad, bt_g)

        
# if __name__ == "__main__":
#     unittest.main()