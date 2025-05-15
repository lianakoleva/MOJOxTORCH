import torch
import inspect

from typing import Sequence, Optional, cast

from pathlib import Path
from dataclasses import dataclass

from max.driver import Tensor, accelerator_count, Accelerator
from max.dtype import DType
from max.engine.api import InferenceSession
from max.engine import Model
from max.graph import (
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
    Shape,
    KernelLibrary,
    Type,
)
from max.mlir import Context
from max._core import graph as _graph
from max.graph import ops

from max import mlir

import matplotlib.pyplot as plt
import os
import max


def display(img):
    plt.imshow(img)


@dataclass(init=False)
class CustomOpLibrary:
    _context: Context
    _kernel_library: KernelLibrary
    _path: Path
    _session: InferenceSession

    def __init__(self, path: Path):
        devices = [] if accelerator_count() == 0 else [Accelerator()]
        # print("__init__devices", devices)
        self._context = Context()
        self._kernel_library = KernelLibrary(self._context, [])
        self._session = InferenceSession(devices=devices)
        self._path = path

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            return object.__getattribute__(self, attr)
        return CustomOp(self, attr)


@dataclass
class CustomOp:
    library: CustomOpLibrary
    name: str


###############################################################################
# Convert torch.Tensor to a TensorType
###############################################################################


def convert_type(dtype: torch.dtype) -> DType:
    table: dict[torch.dtype, DType] = {
        torch.bool: DType.bool,
        torch.float16: DType.float16,
        torch.float32: DType.float32,
        torch.float64: DType.float64,
        torch.int8: DType.int8,
        torch.int16: DType.int16,
        torch.int32: DType.int32,
        torch.int64: DType.int64,
        torch.uint8: DType.uint8,
        torch.uint16: DType.uint16,
        torch.uint32: DType.uint32,
        torch.uint64: DType.uint64,
    }

    return table[dtype]


convert_shape = Shape


def convert_device(device: torch.device) -> DeviceRef:
    type = device.type
    index = device.index or 0
    if type == "cpu":
        # print("cpu")
        return DeviceRef.CPU(index)
    elif type == "cuda":
        # print("cuda")
        return DeviceRef.GPU(index)
    else:
        # print("unknown")
        raise TypeError(f"Unable to convert {type} to a MAX device type.")


def torch_tensor_to_type(tensor: torch.Tensor) -> Type:
    dtype = convert_type(tensor.dtype)
    shape = convert_shape(tensor.shape)
    device = convert_device(tensor.device)
    return TensorType(dtype, shape, device=device)


###############################################################################
# Tensor Conversions
###############################################################################


def to_max_tensor(tensor: torch.Tensor) -> Tensor:
    return Tensor.from_dlpack(tensor)


def to_torch_tensors(result: Sequence[Tensor]) -> torch.Tensor | Sequence[torch.Tensor]:
    torch_tensors = [torch.from_dlpack(t) for t in result]
    return torch_tensors[0] if len(torch_tensors) == 1 else torch_tensors


def custom_op_graph(op: CustomOp, *args, out_like: list[torch.Tensor]) -> Graph:
    input_types = [torch_tensor_to_type(t) for t in args]
    output_types = [torch_tensor_to_type(t) for t in out_like]
    graph_types = [*(t.as_buffer() for t in output_types), *input_types]

    kernel_path = op.library._path

    with Graph(
        op.name, input_types=graph_types, custom_extensions=[kernel_path]
    ) as graph:
        results = ops.custom(
            op.name,
            list(graph.inputs[len(output_types):]),
            out_types=output_types,
        )
        for input, result in zip(graph.inputs, results):
            input.buffer[...] = result

        graph.output()

    return graph


def op_signature(op: mlir.Operation) -> inspect.Signature:
    # TODO: support non-dps outputs
    # for key in op.attributes:
    # print("key = ", key)
    # print("op.attributes ", op.attributes)
    num_dps_outputs = op.attributes["mogg.num_dps_outputs"].value
    io_specs = [attr.value for attr in op.attributes["mogg.args_io_specs"]]
    arg_names = [attr.value for attr in op.attributes["mogg.arg_src_names"]]
    input_specs = io_specs[num_dps_outputs:]
    nargs = len(input_specs)
    arg_names = arg_names[num_dps_outputs : num_dps_outputs + nargs]
    args = [
        inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor
        )
        for name in arg_names
    ]
    result_type = (
        torch.Tensor
        if num_dps_outputs == 1
        else tuple[tuple([torch.Tensor] * num_dps_outputs)]
    )
    return inspect.Signature(args, return_annotation=result_type)


def register_custom_op(op: CustomOp, name: Optional[str] = None):
    # TODO: Figure out how to use all that fancy signature binding stuff.
    # For now we support one input and one output.

    # This will hold the compiled model once the registered fake tensor function
    # is invoked for the first time.
    model: Optional[Model] = None
    registered_fake = None

    kernel = op.library._kernel_library._analysis.kernel(op.name)
    # Not sure what to do with these. Do we expose bindings for KGEN ops?

    # TODO: Why is the smart library loading happening in the Graph constructor?
    graph = Graph("foo", custom_extensions=[op.library._path])
    kernel: mlir.Operation = graph._kernel_library._analysis.kernel(op.name)
    # kernel: kgen.GeneratorOp = analysis.kernel(op.name)
    signature: inspect.Signature = op_signature(kernel)

    # Compile the model if it has not been compiled already.
    def compile_model(*args: torch.Tensor) -> Sequence[torch.Tensor]:
        nonlocal model
        assert (
            registered_fake is not None
        ), "Must register_fake for pytorch custom op before compiling"
        results = registered_fake(*args)

        if model is not None:
            return results

        result_like = [results] if isinstance(results, torch.Tensor) else results
        model = op.library._session.load(
            custom_op_graph(op, *args, out_like=result_like)
        )
        return results

    def custom_op(*args: torch.Tensor) -> torch.Tensor | Sequence[torch.Tensor]:
        # In eager mode, the fake_tensor function will not be called,
        # so we call it here.
        if model is None:
            compile_model(*args)

        assert model is not None

        # registered_fake with real inputs will create buffers for the outputs
        outputs = registered_fake(*args)
        dps = outputs if isinstance(outputs, tuple) else (outputs,)
        converted = map(to_max_tensor, (*dps, *args))
        model(*converted)
        return outputs

    custom_op.__signature__ = signature
    name = name or f"max::torch.{op.name}"
    custom_op = torch.library.custom_op(name, custom_op, mutates_args=())

    @custom_op.register_fake
    def fake_fn(*args: torch.Tensor):
        assert (
            registered_fake is not None
        ), "Must register_fake for pytorch custom op before compiling"
        return compile_model(*args)

    def register_fake(fn):
        nonlocal registered_fake
        registered_fake = fn
        return fn

    custom_op.register_fake = register_fake
    return custom_op