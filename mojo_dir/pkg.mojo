from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList, Index
from layout import LayoutTensor, Layout
from gpu import global_idx
from math import ceildiv, sqrt
from max.math import isclose
from algorithm.functional import vectorize


@register("abs")
struct Abs:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return abs(x.load[width](idx))

        foreach[func, target=target](out, ctx)

@register("max")
struct Max:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = 3],
        ctx: DeviceContextPtr,
    ) raises:
        # TODO: find a mojo-esque way to define this kernel (reduce_max?)
        out[0] = x[Index(0, 0, 0)]
        for i in range(x.shape()[0]):
            for j in range(x.shape()[1]):
                for k in range(x.shape()[2]):
                    out[0] = max(out[0], x[Index(i, j, k)])

@register("allclose")
struct AllClose:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        out: OutputTensor[type = DType.bool],
        input: InputTensor[type = DType.float32, rank = out.rank],
        other: InputTensor[type = DType.float32, rank = out.rank],
        # TODO: figure out how to make optional / register overloaded implementation
        # atol: Float64 = 1e-8,
        # rtol: Float64 = 1e-5,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return isclose(input.load[width](idx), other.load[width](idx), rtol=1e-3, atol=1e-3)

        foreach[func, target=target](out, ctx)

@register("cosine_similarity")
struct CosineSimilarity:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank + 1],
        y: InputTensor[type = out.type, rank = out.rank + 1],
        # TODO: use dim, eps
        dim: Int,
        eps: Float64,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            var dot = (x.load[width](idx) * y.load[width](idx)).reduce_add()
            var x_norm = sqrt((x.load[width](idx) * x.load[width](idx)).reduce_add())
            var y_norm = sqrt((y.load[width](idx) * y.load[width](idx)).reduce_add())
            return dot / (x_norm * y_norm)

        foreach[func, target=target](out, ctx)