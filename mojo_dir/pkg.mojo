from compiler import register
from max.tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
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
        a: InputTensor[type = out.type, rank = out.rank],
        b: InputTensor[type = out.type, rank = out.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[a.rank]) -> SIMD[a.type, width]:
            return max(a.load[width](idx), b.load[width](idx))

        foreach[func, target=target](out, ctx)


@register("allclose")
struct AllClose:
    @staticmethod
    fn execute[
        target: StaticString,
        atol: Float64 = 1e-8, # TODO: check if used correctly
        rtol: Float64 = 1e-5, # TODO: check if used correctly
    ](
        out: OutputTensor[type = DType.bool],
        input: InputTensor[type = DType.float32, rank = out.rank],
        other: InputTensor[type = DType.float32, rank = out.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            return isclose(input.load[width](idx), other.load[width](idx), rtol=rtol, atol=atol)

        foreach[func, target=target](out, ctx)

@register("cosine_similarity")
struct CosineSimilarity:
    @staticmethod
    fn execute[
        target: StaticString,
        dim: Int = 1,
        eps: Float64 = 1e-8,
    ](
        out: OutputTensor,
        x: InputTensor[type = out.type, rank = out.rank + 1],
        y: InputTensor[type = out.type, rank = out.rank + 1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[out.rank]) -> SIMD[out.type, width]:
            #var val_x = x.load[width](idx)
            #var val_y = y.load[width](idx)
            return -1.0

        foreach[func, target=target](out, ctx)