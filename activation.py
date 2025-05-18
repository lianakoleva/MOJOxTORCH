# inspired by max.nn.linear.py
from max.graph import (
    TensorValue,
)
from max.nn.layer import Module
import max.graph.ops

class ReLU(Module):
    """
    Applies a ReLU transformation to incoming data: :math:`y = max(0, x)`.

    Example:

    .. code-block:: python

        self.activation_function = ReLU()

        # Input tensor of shape: [batch, ..., 256]
        input_tensor: TensorValue
        output = self.activation_function(input_tensor)
    """

    # device: DeviceRef
    # """The device where matrix operations are performed."""


    inplace: bool = False
    """Optionally do the operation inplace""" # TODO: is allowed in mojo?=


    def __init__(
        self,
        # device: DeviceRef,
        inplace: bool = False,
    ) -> None:
        """Initializes the linear layer with weights and optional bias.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When :obj:`True`, adds a bias vector to the layer.
                Defaults to :obj:`False`.
        """
        super().__init__()

        # TODO: why does mojo.nn.linear.Linear not complain when device is not passed
        # self.device = device 
        self.inplace = inplace


    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a ReLU transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.
        """
        return max(0, x)


class Softmax(Module):

    # device: DeviceRef
    # """The device where matrix operations are performed."""


    def __init__(
        self,
        # device: DeviceRef,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        # TODO: why does mojo.nn.linear.Linear not complain when device is not passed
        # self.device = device 
        self.inplace = inplace


    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.softmax(0, x)

