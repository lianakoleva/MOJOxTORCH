from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from layout import LayoutTensor, Layout
from gpu import global_idx
from math import ceildiv, sqrt
from algorithm.functional import vectorize

@register("testop")
struct TestOp:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        img_out: OutputTensor[type = DType.uint8, rank=2],
        img_in: InputTensor[type = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            print("cpu: one")
        else:

            var dev_ctx = ctx.get_device_context()
            print("else: two") #, dev_ctx)