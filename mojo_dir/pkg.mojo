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
        in: InputTensor[type = DType.uint8, rank=3],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn absFunc[
            simd_width: Int
        ](idx: IndexList[out.rank]) -> SIMD[DType.uint8, simd_width]:
            var i = idx[0]
            var j = idx[1]
            var k = idx[2]

            var idx = IndexList[3](i, j, k)

            var val = in.load[simd_width](idx).cast[DType.float32]()

            return abs(val).cast[DType.uint8]()

        foreach[absFunc, target=target, simd_width=1](out, ctx)


#   for k in range(in_tens.dim(2)):
#             #             let val = in_tens[i, j, k]
#             #             out_tens[i, j, k] = val if val >= 0 else -val
#         else:
#             # For GPU, we can use vectorized operations
#             # let size = in_tens.num_elements()
#             # @parameter
#             # fn process_element[simd_width: Int](idx: Int):
#             #     let val = in_tens.simd_load[simd_width](idx)
#             #     let abs_val = val if val >= 0 else -val
#             #     out_tens.simd_store[simd_width](idx, abs_val)
            
#             #vectorize[8](size, process_element)

#             @parameter
#             @always_inline
#             fn func[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
#                 if x.load[width](idx) > 0:
#                     return x.load[width](idx)
#                 else:
#                     return -x.load[width](idx)

#             foreach[func, target=target](out, ctx)