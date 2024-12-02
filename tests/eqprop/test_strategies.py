import pytest
import torch

from src.core.eqprop.python import activation, strategy


class TestSecondOrderStrategy:
    """Test the second order strategy."""

    lap = torch.tensor([[4, -2], [-2, 2]]).float()

    def test_laplacian(self, second_order_strategy):
        """Test if the laplacian is computed correctly."""
        st = second_order_strategy
        # Check the laplacian value
        assert torch.allclose(st.laplacian(), self.lap)

    @pytest.mark.parametrize("v", [-0.5, 1.0])
    def test_jacobian(self, second_order_strategy, v):
        """Test if the jacobian is computed correctly."""
        expected_j = 1.0 if v == 1 else 0.0
        v = torch.tensor([[v, v]])
        st = second_order_strategy
        J1 = st.jacobian(v)
        assert len(J1.shape) == 3
        assert torch.allclose(J1, self.lap + torch.tensor([[expected_j, 0.0], [0.0, 0.0]]))

    @pytest.mark.parametrize("x", [[1.0, -1.0], [1.0, 1.0]])
    def test_rhs(self, second_order_strategy, x):
        """Test if the right hand side is computed correctly."""
        if x == [1.0, -1.0]:
            expected_rhs = torch.tensor([[1.0, 0.0]])
        elif x == [1.0, 1.0]:
            expected_rhs = torch.tensor([[-1.0, 0.0]])
        else:
            raise ValueError("Invalid x")
        x = torch.tensor([x])
        st = second_order_strategy
        st.reset()
        assert torch.allclose(st.rhs(x), expected_rhs)

    @pytest.mark.parametrize("v, x", [(-0.5, [1.0, -1.0]), (0.5, [1.0, 1.0])])
    def test_residual(self, second_order_strategy, v, x):
        """Test if the residual is computed correctly."""
        st = second_order_strategy
        st.reset()
        assert torch.allclose(
            st.residual(v=torch.tensor([[v, v]]), x=torch.tensor([x]), i_ext=None),
            torch.tensor([[0.0, 0.0]]),
        )

    @pytest.mark.parametrize("v, x", [(-0.5, [1.0, -1.0]), (0.5, [1.0, 1.0])])
    def test_lin_solve(self, second_order_strategy, v, x):
        """Test if the linear solve without activation is computed
        correctly."""
        st = second_order_strategy
        st.reset()
        assert torch.allclose(st.lin_solve(torch.tensor([x]), None), torch.tensor([[v, v]]))


@pytest.mark.parametrize(
    "x, v",
    [
        ([1.0, -1.0], [-0.5, -0.5]),
        ([1.0, 1.0], [0.5, 0.5]),
        ([-1.0, -1.0], [-1.2, -1.2]),
    ],
)
@pytest.mark.parametrize("strategy_name", ["NewtonStrategy"])
def test_strategy_solve(toymodel, strategy_name, x, v):
    """Test if the strategy can solve the system."""
    # Define the input
    st: strategy.AbstractStrategy = getattr(strategy, strategy_name)(
        activation=activation.SymReLU(Vl=-0.6, Vr=0.6),
        max_iter=20,
        clip_threshold=1,
        add_nonlin_last=False,
        atol=1e-6,
    )
    # st.set_strategy_params(toymodel)
    st.reset()
    x = torch.tensor([x])
    v = torch.tensor([v])
    v_pred = torch.cat(st.solve(x, None))
    assert torch.allclose(v_pred, v)


# def test_EqPropSolver_energy():
#     # Define the solver
#     dims = [2, 3, 1]
#     strategy = TorchStrategy()
#     activation = eqprop_utils.OTS()
#     solver = EqPropSolver(strategy, activation)
#     solver.set_dims(dims)

#     # Define the input and nodes
#     x = torch.tensor([[1.0, 2.0]])
#     nodes = [
#         torch.tensor([[0.5, 0.2, 0.3]]),
#         torch.tensor([[0.1]]),
#     ]

#     # Compute the energy
#     energy = solver.energy(nodes, x)

#     # Check the energy value
#     assert torch.isclose(energy, torch.tensor([0.1545]), rtol=1e-4)

# def test_EqPropSolver_solve():
#     # Define the solver
#     dims = [2, 3, 1]
#     strategy = TorchStrategy()
#     activation = eqprop_utils.OTS()
#     solver = AnalogEqPropSolver(strategy, activation)
#     solver.set_dims(dims)

#     # Define the input
#     x = torch.tensor([[1.0, 2.0]])

#     # Solve for the equilibrium point
#     nodes = solver.solve(x, 0)

#     # Check the shape of the nodes
#     assert len(nodes) == len(dims) - 1
#     for i in range(len(nodes)):
#         assert nodes[i].shape == (1, dims[i+1])
