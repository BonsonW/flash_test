#include <torch/torch.h>
#include <ATen/Functions.h>
#include <optional>
#include <math.h>

torch::Tensor construct_local_mask(
    int seqlen_q,
    int seqlen_k,
    int win_l,
    int win_r,
    torch::Device device
) {
    auto opts = torch::TensorOptions(torch::kInt64);
    auto row_idx = torch::arange(seqlen_q, opts).reshape({seqlen_q, 1}).to(device);
    auto col_idx = torch::arange(seqlen_k, opts).to(device);
    auto sq = seqlen_q;
    auto sk = torch::full_like(col_idx, seqlen_k).to(device);
    return torch::logical_or(
        col_idx > torch::minimum(row_idx + sk - sq + win_r, sk),
        col_idx < row_idx + sk - sq - win_l
    );
}

torch::Tensor attention_ref(
    torch::Tensor &q,
    torch::Tensor &k,
    torch::Tensor &v,
    int win_l,
    int win_r
) {
    torch::Tensor local_mask;
    auto seqlen_q = q.size(1);
    auto seqlen_k = k.size(1);
    auto d = q.size(-1);
    auto scores = torch::einsum("bthd,bshd->bhts", {q / std::sqrt(d), k});
    if (win_l >= 0 || win_r >= 0) {
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            win_l,
            win_r,
            scores.device()
        );
        scores.masked_fill_(local_mask, -INFINITY);
    }
        
    auto attention = torch::softmax(scores, -1);
    if (win_l >= 0 || win_r >= 0) {
        attention = attention.masked_fill(torch::all(local_mask, -1, true), 0.0);
    }
    return torch::einsum("bhts,bshd->bthd", {attention, v});
}

int main(int argc, char* argv[]) {
    auto device_idx = 0;
    auto batchsize = 64;
    auto seqlen = 64;
    auto nhead = 8;
    auto headdim = 64;
    auto win_l = 16;
    auto win_r = 16;
    auto softmax_scale = 1.0 / std::sqrt(headdim);

    torch::InferenceMode guard;
    auto device = torch::Device(torch::DeviceType::CUDA, device_idx);
    c10::DeviceGuard device_guard(device);

    auto q = torch::randn({batchsize, seqlen, nhead, headdim}).to(device).to(torch::kHalf);
    auto k = torch::randn({batchsize, seqlen, nhead, headdim}).to(device).to(torch::kHalf);
    auto v = torch::randn({batchsize, seqlen, nhead, headdim}).to(device).to(torch::kHalf);

    auto ref_out = attention_ref(
        q, k, v,
        win_l, win_r
    );
    
    auto flash_res = at::_flash_attention_forward(
        q, k, v,
        std::nullopt, std::nullopt, // cum_seq_q, cum_seq_k
        q.size(1), k.size(1), // max_seqlen_q, max_seqlen_k
        0.0, // dropout
        false, // casual
        false, // return debug mask
        softmax_scale,
        win_l, win_r,
        std::nullopt, // seqused k
        std::nullopt // alibi slopes
    );
    auto flash_out = std::get<0>(flash_res);

    auto all_close = torch::allclose(ref_out, flash_out, 0.0, 1e-02);

    if (all_close) {
        fprintf(stderr, "test passed\n");
    } else {
        fprintf(stderr, "test failed\n");
    }
    
    return 0;
}