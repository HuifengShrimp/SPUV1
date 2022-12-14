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

#include "spu/mpc/beaver/trusted_party.h"

#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {
namespace {

enum class RecOp : uint8_t {
  ADD = 0,
  XOR = 1,
};

// reconstruct P0's data &
std::pair<std::vector<ArrayRef>, std::vector<ArrayRef>> reconstruct(
    RecOp op, absl::Span<const PrgSeed> seeds,
    absl::Span<const PrgArrayDesc> descs) {
  std::vector<ArrayRef> r0(descs.size());
  std::vector<ArrayRef> rs(descs.size());

  for (size_t rank = 0; rank < seeds.size(); rank++) {
    for (size_t idx = 0; idx < descs.size(); idx++) {
      // auto t = prgReplayArray(seeds[rank], descs[idx]);
      auto t = prgReplayArray(0, descs[idx]);
      if (rank == 0) {
        r0[idx] = t;
        rs[idx] = t.clone();
      } else {
        if (op == RecOp::ADD) {
          ring_add_(rs[idx], t);
        } else if (op == RecOp::XOR) {
          ring_xor_(rs[idx], t);
        } else {
          YASL_ENFORCE("not supported reconstruct op");
        }
      }
    }
  }

  return {r0, rs};
}

void checkDescs(absl::Span<const PrgArrayDesc> descs) {
  for (size_t idx = 1; idx < descs.size(); idx++) {
    YASL_ENFORCE(descs[0].field == descs[idx].field);
    YASL_ENFORCE(descs[0].numel == descs[idx].numel);
  }
}

}  // namespace

void TrustedParty::setSeed(size_t rank, size_t world_size,
                           const PrgSeed& seed) {
  YASL_ENFORCE(rank < world_size,
               "rank={} should be smaller then world_size={}", rank,
               world_size);

  std::unique_lock lock(seeds_mutex_);

  if (seeds_.size() == 0) {
    seeds_.resize(world_size);
    seeds_[rank] = seed;
  } else {
    YASL_ENFORCE(world_size == seeds_.size(),
                 "parties claim different world_size, prev={}, cur={}",
                 seeds_.size(), world_size);

    YASL_ENFORCE(!seeds_[rank].has_value() || seeds_[rank].value() == seed);

    seeds_[rank] = seed;
  }
}

std::vector<PrgSeed> TrustedParty::getSeeds() const {
  std::unique_lock lock(seeds_mutex_);

  std::vector<PrgSeed> seeds;

  for (size_t rank = 0; rank < seeds_.size(); rank++) {
    YASL_ENFORCE(seeds_[rank].has_value(), "seed for rank={} not set", rank);
    seeds.push_back(seeds_[rank].value());
    // std::cout<<"-------seeds--------"<<std::endl;
    // std::cout<<seeds_[rank].value()<<std::endl;
  }

  return seeds;
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


ArrayRef TrustedParty::adjustMul(absl::Span<const PrgArrayDesc> descs) {
  YASL_ENFORCE_EQ(descs.size(), 3u);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);


  // r0[2] += rs[0] * rs[1] - rs[2];
  ring_add_(r0[2], ring_sub(ring_mul(rs[0], rs[1]), rs[2]));
  return r0[2];
}

ArrayRef TrustedParty::adjustDot(absl::Span<const PrgArrayDesc> descs, size_t M,
                                 size_t N, size_t K) {
  YASL_ENFORCE_EQ(descs.size(), 3u);
  YASL_ENFORCE(descs[0].numel == M * K);
  YASL_ENFORCE(descs[1].numel == K * N);
  YASL_ENFORCE(descs[2].numel == M * N);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[2] += rs[0] dot rs[1] - rs[2];

  //1024-lj
  // auto m1 = ring_ones(descs[0].field, N * N);
  // ArrayRef m1T(makeType<RingTy>(descs[0].field), N * N);
  // for (size_t i = 0 ; i < N * N ; i++) {
  //   size_t index = (i % N) * N + (i / N);
  //   m1T.at<int32_t>(index) = m1.at<int32_t>(i);
  // }
  // r0[2] = ring_mmul(r0[2], m1T, M, N, N);

  ring_add_(r0[2], ring_sub(ring_mmul(rs[0], rs[1], M, N, K), rs[2]));
  return r0[2];
}

//lj
std::tuple<ArrayRef, ArrayRef, ArrayRef, ArrayRef, ArrayRef> TrustedParty::adjustLR(absl::Span<const PrgArrayDesc> descs, 
                                                                                          size_t M,size_t N, size_t K) {
  YASL_ENFORCE_EQ(descs.size(), 8u);

  YASL_ENFORCE(descs[0].numel == M * N);
  YASL_ENFORCE(descs[1].numel == K * N);
  YASL_ENFORCE(descs[2].numel == M * K);
  YASL_ENFORCE(descs[3].numel == K * M);
  YASL_ENFORCE(descs[4].numel == N * M * N);
  YASL_ENFORCE(descs[5].numel == K * N);
  YASL_ENFORCE(descs[6].numel == K * N);
  YASL_ENFORCE(descs[7].numel == N * N);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);

  //1024-lj-transpose
  auto r1T = rs[0].clone();
  TransposeInplace(r1T, M, N);

  size_t i, j;
  ring_add_(r0[3], ring_sub(ring_mmul(rs[1], r1T, K, M, N), rs[3]));

  // using U = ring2k_t;
  SimdTrait<ArrayRef>::PackInfo pi;
  std::vector<ArrayRef> vec_c2;

  for(i = 0; i < M * N; i++) {
    ArrayRef c = r1T.slice(i, i+1);
    for(j = 0; j < K * N; j++) {
      ArrayRef d = rs[1].slice(j, j+1);
      vec_c2.push_back(ring_mul(c, d));
    }
  }

  ArrayRef c2 = SimdTrait<ArrayRef>::pack(vec_c2.begin(), vec_c2.end(), pi);

  ring_add_(r0[4], ring_sub(c2, rs[4]));

  ring_add_(r0[5], ring_sub(ring_mmul(ring_mmul(rs[1], r1T, K, M, N), rs[0], K, N, M), rs[5]));

  auto r3T = rs[2].clone();
  TransposeInplace(r3T, K, M);

  ring_add_(r0[6], ring_sub(ring_mmul(r3T, rs[0], K, N, M), rs[6]));

  ring_add_(r0[7], ring_sub(ring_mmul(r1T, rs[0], N, N, M), rs[7]));

  return {r0[3], r0[4], r0[5], r0[6], r0[7]};
}

ArrayRef TrustedParty::adjustAnd(absl::Span<const PrgArrayDesc> descs) {
  YASL_ENFORCE_EQ(descs.size(), 3u);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::XOR, getSeeds(), descs);
  // r0[2] ^= (rs[0] & rs[1]) ^ rs[2];
  ring_xor_(r0[2], ring_xor(ring_and(rs[0], rs[1]), rs[2]));
  return r0[2];
}

ArrayRef TrustedParty::adjustTrunc(absl::Span<const PrgArrayDesc> descs,
                                   size_t bits) {
  YASL_ENFORCE_EQ(descs.size(), 2u);
  checkDescs(descs);

  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), descs);
  // r0[1] += (rs[0] >> bits) - rs[1];
  ring_add_(r0[1], ring_sub(ring_arshift(rs[0], bits), rs[1]));
  return r0[1];
}

ArrayRef TrustedParty::adjustRandBit(const PrgArrayDesc& desc) {
  auto [r0, rs] = reconstruct(RecOp::ADD, getSeeds(), absl::MakeSpan(&desc, 1));
  YASL_ENFORCE(r0.size() == 1 && rs.size() == 1);

  // r0[0] += bitrev - rs[0];
  ring_add_(r0[0], ring_sub(ring_randbit(desc.field, desc.numel), rs[0]));
  return r0[0];
}

}  // namespace spu::mpc
