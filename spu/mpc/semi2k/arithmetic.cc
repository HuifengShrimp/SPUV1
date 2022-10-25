// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "spu/mpc/semi2k/arithmetic.h"

#include "spu/core/profile.h"
#include "spu/core/vectorize.h"
#include "spu/mpc/common/abprotocol.h"  // zero_a
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/semi2k/object.h"
#include "spu/mpc/semi2k/type.h"
#include "spu/mpc/util/communicator.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc::semi2k {

ArrayRef ZeroA::proc(KernelEvalContext* ctx, FieldType field,
                     size_t size) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, size);

  auto* prg_state = ctx->caller()->getState<PrgState>();

  auto [r0, r1] = prg_state->genPrssPair(field, size);
  return ring_sub(r0, r1).as(makeType<AShrTy>(field));
}

ArrayRef P2A::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();

  auto x = zero_a(ctx->caller(), field, in.numel());

  if (comm->getRank() == 0) {
    ring_add_(x, in);
  }

  return x.as(makeType<AShrTy>(field));
}

ArrayRef A2P::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  const auto field = in.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto out = comm->allReduce(ReduceOp::ADD, in, kBindName);
  return out.as(makeType<Pub2kTy>(field));
}

ArrayRef NotA::proc(KernelEvalContext* ctx, const ArrayRef& in) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in);

  auto* comm = ctx->caller()->getState<Communicator>();

  // First, let's show negate could be locally processed.
  //   let X = sum(Xi)     % M
  //   let Yi = neg(Xi) = M-Xi
  //
  // we get
  //   Y = sum(Yi)         % M
  //     = n*M - sum(Xi)   % M
  //     = -sum(Xi)        % M
  //     = -X              % M
  //
  // 'not' could be processed accordingly.
  //   not(X)
  //     = M-1-X           # by definition, not is the complement of 2^k
  //     = neg(X) + M-1
  //
  auto res = ring_neg(in);
  if (comm->getRank() == 0) {
    const auto field = in.eltype().as<Ring2k>()->field();
    ring_add_(res, ring_not(ring_zeros(field, in.numel())));
  }

  return res.as(in.eltype());
}

////////////////////////////////////////////////////////////////////
// add family
////////////////////////////////////////////////////////////////////
ArrayRef AddAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  YASL_ENFORCE(lhs.numel() == rhs.numel());
  auto* comm = ctx->caller()->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_add(lhs, rhs).as(lhs.eltype());
  }

  return lhs;
}

ArrayRef AddAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  YASL_ENFORCE(lhs.numel() == rhs.numel());
  YASL_ENFORCE(lhs.eltype() == rhs.eltype());

  return ring_add(lhs, rhs).as(lhs.eltype());
}

////////////////////////////////////////////////////////////////////
// multiply family
////////////////////////////////////////////////////////////////////
ArrayRef MulAP::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  return ring_mul(lhs, rhs).as(lhs.eltype());
}

ArrayRef MulAA::proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                     const ArrayRef& rhs) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);

  const auto field = lhs.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();
  auto [a, b, c] = beaver->Mul(field, lhs.numel());

  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(lhs, a), ring_sub(rhs, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kBindName);
      });

  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  // Zi = Ci + (X - A) * Bi + (Y - B) * Ai + <(X - A) * (Y - B)>
  auto z = ring_add(ring_add(ring_mul(x_a, b), ring_mul(y_b, a)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mul(x_a, y_b));
  }

  return z.as(lhs.eltype());
}

//1024-lj
static void TransposeInplace(ArrayRef mat, size_t nrows, size_t ncols) {
  YASL_ENFORCE_EQ((size_t)mat.numel(), nrows * ncols);
  const auto field = mat.eltype().as<Ring2k>()->field();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto xmat = xt_mutable_adapt<ring2k_t>(mat);
    xmat.reshape({nrows, ncols});
    auto xmatT = xt::eval(xt::transpose(xmat));
    std::copy_n(xmatT.begin(), xmatT.size(), xmat.data());
  });
}

////////////////////////////////////////////////////////////////////
// matmul family
////////////////////////////////////////////////////////////////////
ArrayRef MatMulAP::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);
  return ring_mmul(x, y, M, N, K).as(x.eltype());
}

ArrayRef MatMulAA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, y);

  std::cout<<"----M, N, K----"<<std::endl;
  std::cout<<M<<std::endl;
  std::cout<<N<<std::endl;
  std::cout<<K<<std::endl;

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

  // generate beaver multiple triple.
  auto [a, b, c] = beaver->Dot(field, M, N, K);

  // Open x-a & y-b
  auto res =
      vectorize({ring_sub(x, a), ring_sub(y, b)}, [&](const ArrayRef& s) {
        return comm->allReduce(ReduceOp::ADD, s, kBindName);
      });
  auto x_a = std::move(res[0]);
  auto y_b = std::move(res[1]);

  //lj
  auto iden2 = ring_ones(field, K * K);
  TransposeInplace(iden2, K, K);
  x_a = ring_mmul(x_a, iden2, M, K, K);


  // Zi = Ci + (X - A) dot Bi + Ai dot (Y - B) + <(X - A) dot (Y - B)>
  auto z = ring_add(
      ring_add(ring_mmul(x_a, b, M, N, K), ring_mmul(a, y_b, M, N, K)), c);
  if (comm->getRank() == 0) {
    // z += (X-A) * (Y-B);
    ring_add_(z, ring_mmul(x_a, y_b, M, N, K));
  }
  return z.as(x.eltype());
}

////////////////////////////////////////////////////////////////////
// lj : logreg family
////////////////////////////////////////////////////////////////////
ArrayRef LogRegAP::proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& w,
                        const ArrayRef& y, size_t M, size_t N, size_t K) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, w, y);
  
  const auto field = x.eltype().as<Ring2k>()->field();
  size_t i = 0, index;
  ArrayRef x_(makeType<RingTy>(field), N * M);
  for ( i = 0 ; i < x.elsize() ; i++){
    index = (i % N) * M + (i / N);
    x_.at<int32_t>(index) = x.at<int32_t>(i);
  }

  auto y_pred = ring_mmul(w, x_, 1, M, N);

  auto err = ring_sub(y_pred, y);

  auto grad = ring_mmul(err, x, 1, N, M);

  return grad.as(x.eltype());
}

ArrayRef LogRegAA::proc(KernelEvalContext* ctx, const ArrayRef& x, const ArrayRef& w,
                        const ArrayRef& y, size_t M, size_t K, size_t N) const {

  SPU_PROFILE_TRACE_KERNEL(ctx, x, w, y);

  std::cout<<"-------M, N, K---------"<<std::endl;
  std::cout<<M<<std::endl;
  std::cout<<N<<std::endl;
  std::cout<<K<<std::endl;

  const auto field = x.eltype().as<Ring2k>()->field();
  auto* comm = ctx->caller()->getState<Communicator>();
  auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();


  // generate correlated randomness.
  auto [r1, r2, r3, c1, c2, c3, c4, c5] = beaver->lr(field, M, N, K);

  auto x_r1 = comm->allReduce(ReduceOp::ADD, ring_sub(x, r1), kBindName);

  auto w_r2 = comm->allReduce(ReduceOp::ADD, ring_sub(w, r2), kBindName);

  auto y_r3 = comm->allReduce(ReduceOp::ADD, ring_sub(y, r3), kBindName);

  size_t i = 0, j = 0, k = 0;

  //1024-the transpose of x_r1
  auto x_r1T = x_r1.clone();
  TransposeInplace(x_r1T, M, N);

  //1024-the transpose of r1
  auto r1T = r1.clone();
  TransposeInplace(r1T, M, N);
  
  SimdTrait<ArrayRef>::PackInfo pi;
  std::vector<ArrayRef> vec_c2;

  //1024-the computation of r2(x')Tr1
  k = 0;
  for(i = 0; i < N; i++) {
    ArrayRef a = ring_zeros(field, 1);
    for(j = 0; j < M * N; j++) {
      ArrayRef b = x_r1.slice(j, j+1);
      ring_add_(a, ring_mul(c2.slice(k, k+1), b));
      k++;
    }
    vec_c2.push_back(a);
  }
  
  ArrayRef c2_ = SimdTrait<ArrayRef>::pack(vec_c2.begin(), vec_c2.end(), pi);


  auto iden2 = ring_twos(field, K * M);

  //1024-lj-transpose
  auto y_r3T = y_r3.clone();
  TransposeInplace(y_r3T, M, K);

  auto r3T = r3.clone();
  TransposeInplace(r3T, M, K);

  auto tmp = ring_add(ring_add(
  ring_mmul(y_r3T, r1, K, N, M),
  ring_mmul(r3T, x_r1, K, N, M)),
  c4);

  ArrayRef four = ring_four(field);

  std::vector<ArrayRef> vec_tmp;

  for(i = 0; i < K * N; i++) {
    ArrayRef b = tmp.slice(i, i+1);
    vec_tmp.push_back(ring_mul(four, b));
  }
  tmp = SimdTrait<ArrayRef>::pack(vec_tmp.begin(), vec_tmp.end(), pi);


  auto grad  = ring_sub(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(ring_add(
  ring_mmul(iden2, r1, K, N, M),
  ring_mmul(ring_mmul(w_r2, r1T, K, M, N), x_r1, K, N, M)),
  ring_mmul(ring_mmul(r2, x_r1T, K, M, N), x_r1, K, N, M)),
  ring_mmul(c1, x_r1, K, N, M)),
  ring_mmul(ring_mmul(w_r2, x_r1T, K, M, N), r1, K, N, M)),
  ring_mmul(w_r2, c5, K, N, N)),
  c2_),
  c3),
  tmp);

  if (comm->getRank() == 0) {
    auto tmp1 = ring_mmul(ring_mmul(w_r2, x_r1T, K, M, N), x_r1, K, N, M);
    auto tmp2 = ring_mmul(y_r3, x_r1, K, N, M);
    auto tmp3 = ring_mmul(iden2, x_r1, K, N, M);

    std::vector<ArrayRef> vec_tmp2;
    for(i = 0; i < K * N; i++) {
      ArrayRef b = tmp2.slice(i, i+1);
      vec_tmp2.push_back(ring_mul(four, b));
    }
    tmp2 = SimdTrait<ArrayRef>::pack(vec_tmp2.begin(), vec_tmp2.end(), pi);
 
    ring_add_(tmp1, tmp2);
    ring_add_(tmp1, tmp3);
    ring_add_(grad, tmp1);
  }

  return grad.as(x.eltype());
}

ArrayRef LShiftA::proc(KernelEvalContext* ctx, const ArrayRef& in,
                       size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);

  const auto field = in.eltype().as<Ring2k>()->field();
  bits %= SizeOf(field) * 8;
  std::cout<<"----------arithmetic.cc : LShiftA-----------"<<std::endl;
  return ring_lshift(in, bits).as(in.eltype());
}

ArrayRef TruncPrA::proc(KernelEvalContext* ctx, const ArrayRef& x,
                        size_t bits) const {
  SPU_PROFILE_TRACE_KERNEL(ctx, x, bits);
  auto* comm = ctx->caller()->getState<Communicator>();

  // TODO: add trunction method to options.
  if (comm->getWorldSize() == 2u) {
    // SecurlML, local trunction.
    // Ref: Theorem 1. https://eprint.iacr.org/2017/396.pdf
    //lj-todo: truncation
    std::cout<<"----------arithmetic.cc : TruncPrA-----------"<<std::endl;
    return ring_arshift(x, bits).as(x.eltype());
  } else {
    // ABY3, truncation pair method.
    // Ref: Section 5.1.2 https://eprint.iacr.org/2018/403.pdf
    auto* beaver = ctx->caller()->getState<Semi2kState>()->beaver();

    const auto field = x.eltype().as<Ring2k>()->field();
    const auto& [r, rb] = beaver->Trunc(field, x.numel(), bits);

    // open x - r
    auto x_r = comm->allReduce(ReduceOp::ADD, ring_sub(x, r), kBindName);
    auto res = rb;
    if (comm->getRank() == 0) {
      ring_add_(res, ring_arshift(x_r, bits));
    }

    // res = [x-r] + [r], x which [*] is truncation operation.
    return res.as(x.eltype());
  }
}

}  // namespace spu::mpc::semi2k
