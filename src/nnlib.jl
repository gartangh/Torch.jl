using NNlib
using NNlib: PoolDims
using NNlib: ConvDims

function NNlib.conv(x::Tensor{xT, N}, w::Tensor{T,N}, b::Tensor{T}, cdims::ConvDims{M,K,C_in,C_out,S,P,D,F,G}) where {xT,N,T, M,K,C_in,C_out,S,P,D,F,G}
  conv2d(x, w, b, stride = collect(S), padding = [P[1];P[3]], dilation = collect(D), groups = G)
end

function NNlib.conv(x::Tensor, w::Tensor, cdims::ConvDims)
  b = zeros(Tensor{Float32}, size(w)[end], dev = on(w))
  conv(x, w, b, cdims)
end

function NNlib.relu(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_relu(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.leakyrelu(t::Tensor{T,N}) where {T,N}
  ptr = Ref(Ptr{Cvoid}())

  atg_leaky_relu(ptr, t.ptr)
  Tensor{T,N}(ptr[], on(t))
end

function NNlib.softmax(t::Tensor{T,N}; dims = 1) where {T,N}
  _softmax(t, N - dims, options[T])
end

function NNlib.meanpool(t::Tensor, pdims::PoolDims{N,K,S,P,D}) where {N,K,S,P,D}
  ks = collect(NNlib.kernel_size(pdims))
  stride = collect(S)
  pad = [P[1];P[3]]
  op_sz = NNlib.output_size(pdims)

  _meanpool(t, ks, stride, pad, op_sz)
end

function NNlib.maxpool(t::Tensor, pdims::PoolDims{N,K,S,P,D}) where {N,K,S,P,D}
  _maxpool(t, pdims)
end

include("grads.jl")
