"""Per-token torch.compile attention with CUDA streams.
Removes batch dimension — each token processed independently on its own stream."""
import math
import torch

torch.backends.cuda.matmul.allow_tf32 = False

# Pre-allocate streams (max T=8)
_streams = [torch.cuda.Stream() for _ in range(8)]


@torch.compile
def compute_attention_single(qn, qp, Kc, Kp, mask, sm_scale, output, lse):
    """Single-token attention — no batch dimension.

    Inputs:
        qn:       [16, 512]    bf16
        qp:       [16, 64]     bf16
        Kc:       [2048, 512]  bf16
        Kp:       [2048, 64]   bf16
        mask:     [2048]       bool  — True for INVALID
        sm_scale: float

    Outputs (written in-place):
        output: [16, 512]  bf16
        lse:    [16]       f32
    """
    # Score matmuls: [16, 512] @ [512, 2048] + [16, 64] @ [64, 2048] → [16, 2048] f32
    logits = torch.mm(qn, Kc.T, out_dtype=torch.float32) + torch.mm(qp, Kp.T, out_dtype=torch.float32)

    logits.masked_fill_(mask.unsqueeze(0), float('-inf'))
    logits_scaled = logits * sm_scale

    lse.copy_(torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0))
    attn = torch.softmax(logits_scaled, dim=-1)

    output.copy_(torch.mm(attn, Kc.float()).to(torch.bfloat16))


@torch.compile
def prepare_indices_single(sparse_indices):
    """Single-row mask + safe indices."""
    mask = sparse_indices == -1
    safe_indices = sparse_indices.clamp(min=0).long()
    return mask, safe_indices


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse, is_profiling=False):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    # Flatten paged KV cache
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe)

    for t in range(num_tokens):
        stream = _streams[t]
        with torch.cuda.stream(stream):
            mask, safe_idx = prepare_indices_single(sparse_indices[t])
            Kc = Kc_all[safe_idx]
            Kp = Kp_all[safe_idx]
            compute_attention_single(
                q_nope[t], q_pe[t], Kc, Kp, mask, sm_scale, output[t], lse[t]
            )

    # Wait for all streams
    main_stream = torch.cuda.current_stream()
    for t in range(num_tokens):
        main_stream.wait_stream(_streams[t])


if __name__ == "__main__":
    from pathlib import Path
    from cook import check_correctness, benchmark
    from ref import run as ref_fn

    ROOT = Path(__file__).parent.parent
    CONTEST = ROOT.parent / "flashinfer26dsa" / "mlsys26-contest"
    JSONL = str(CONTEST / "workloads" / "dsa_paged" / "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.jsonl")

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print("=== Correctness ===")
    check_correctness(run, ref_fn, jsonl_path=JSONL)
    print("\n=== Benchmark ===")
    benchmark(run, ref_fn, jsonl_path=JSONL)
