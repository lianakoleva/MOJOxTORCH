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

@register("abs")
struct Abs:
    @staticmethod
    fn execute[
        # The kind of device this is running on: "cpu" or "gpu"
        target: StaticString,
    ](
        out: OutputTensor[type = DType.uint8, rank=3],
        x: InputTensor[type = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn absFunc[
            simd_width: Int
        ](idxs: IndexList[out.rank]) -> SIMD[DType.uint8, simd_width]:
            var i = idxs[0]
            var j = idxs[1]
            var k = idxs[2]

            var idx = IndexList[3](i, j, k)

            var val = x.load[simd_width](idx).cast[DType.float32]()

            return abs(val).cast[DType.uint8]()

        foreach[absFunc, target=target, simd_width=1](out, ctx)

@register("max")
struct Max:
    @staticmethod
    fn execute[
        # The kind of device this is running on: "cpu" or "gpu"
        target: StaticString,
    ](
        out: OutputTensor[type = DType.uint8, rank=3],
        a: InputTensor[type = DType.uint8, rank=3],
        b: InputTensor[type = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn maxFunc[
            simd_width: Int
        ](idxs: IndexList[out.rank]) -> SIMD[DType.uint8, simd_width]:
            var i = idxs[0]
            var j = idxs[1]
            var k = idxs[2]

            var idx = IndexList[3](i, j, k)

            var val_a = a.load[simd_width](idx).cast[DType.float32]()
            var val_b = b.load[simd_width](idx).cast[DType.float32]()

            return max(val_a, val_b).cast[DType.uint8]()

        foreach[maxFunc, target=target, simd_width=1](out, ctx)

