#!/usr/bin/env python3
"""6-step × 3-column diagram: DeepSpeed stock vs Tutel-patched MoE dispatch.

Columns:
  Col 0 — Step label (stage name + step number)
  Col 1 — DeepSpeed stock implementation
  Col 2 — Tutel-patched implementation

6 steps (rows):
  1  Routing          noisy gate + top2gating → dense [S,E,C] masks
  2  Dispatch/Encode  dense einsum "sec,sm→ecm"   vs  sparse encode kernel
  3  AllToAll #1      identical (unchanged)
  4  Expert compute   identical (unchanged)
  5  AllToAll #2      identical (unchanged)
  6  Combine/Decode   dense einsum "sec,ecm→sm"   vs  sparse decode kernel
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(__file__))
from excalib import (Diagram, C_DIM, C_TXT, C_OP, C_IDX,
                     FS_DIM, FS_NAME, FS_TITLE, FS_NOTE, FS_BIG, SECTION_GAP)

d = Diagram()

# ── Layout constants ──────────────────────────────────────────────────
LEFT        = 60
COL_LABEL_W = 130       # col 0: step label
GAP         = 20
COL_DS_W    = 520       # col 1: DeepSpeed
COL_TU_W    = 520       # col 2: Tutel

COL0_X = LEFT
COL1_X = COL0_X + COL_LABEL_W + GAP
COL2_X = COL1_X + COL_DS_W + GAP

TOTAL_W = COL2_X + COL_TU_W + LEFT

ROW_H   = 140           # height of each step row
ROW_GAP = 14

# ── Colors ────────────────────────────────────────────────────────────
C_STEP   = "#1971c2"; BG_STEP  = "#dbe4ff"   # step label
C_DS     = "#c92a2a"; BG_DS    = "#ffe3e3"   # DeepSpeed (red = hot/slow)
C_TU     = "#2f9e44"; BG_TU    = "#d3f9d8"   # Tutel (green = fast)
C_SAME   = "#5f3dc4"; BG_SAME  = "#e5dbff"   # identical in both (purple)
C_HDR    = "#1e1e1e"; BG_HDR   = "#f8f9fa"   # header row

def diag_arrow(x1, y1, x2, y2, color="#1e1e1e"):
    dx, dy = x2 - x1, y2 - y1
    d.elements.append({
        "id": d._uid(), "type": "arrow",
        "x": x1, "y": y1, "width": abs(dx), "height": abs(dy),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None,
        "index": d._next_index(), "roundness": {"type": 2},
        "seed": random.randint(1, 2**31), "version": 1,
        "versionNonce": random.randint(1, 2**31),
        "isDeleted": False, "boundElements": None,
        "updated": 1773529200000, "link": None, "locked": False,
        "points": [[0, 0], [dx, dy]],
        "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": "arrow",
        "elbowed": False,
    })

def cell(x, y, w, h, color, bg, fill, title, lines, title_fs=FS_NOTE+2, line_fs=FS_NOTE):
    """Draw one table cell with a bold title and bullet lines."""
    d.rect(x, y, w, h, color, bg, fill, roundness=4)
    ty = y + 8
    d.text(x + 10, ty, title, title_fs, color)
    ty += title_fs + 6
    for line in lines:
        d.text(x + 10, ty, line, line_fs, C_DIM)
        ty += line_fs + 5

def step_label(x, y, w, h, num, name, sub):
    d.rect(x, y, w, h, C_STEP, BG_STEP, "solid", roundness=4)
    d.text(x + 8, y + 10, f"Step {num}", FS_DIM + 1, C_STEP)
    d.text(x + 8, y + 26, name, FS_NOTE + 2, C_TXT)
    d.text(x + 8, y + 46, sub, FS_DIM + 1, C_DIM)

# ━━ TITLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
y = 0
d.text(LEFT, y,
       "DeepSpeed MoE stock  vs  Tutel-patched  —  6 stages, 3 columns",
       FS_BIG, C_TXT)
d.text(LEFT, y + 40,
       "Red = DeepSpeed stock path  ·  Green = Tutel replacement  ·  Purple = identical in both",
       FS_NOTE + 2, C_DIM)
y += 84

# ━━ HEADER ROW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HDR_H = 36
d.rect(COL0_X, y, COL_LABEL_W, HDR_H, C_HDR, BG_HDR, "solid")
d.text(COL0_X + 10, y + 10, "Stage", FS_TITLE - 4, C_TXT)
d.rect(COL1_X, y, COL_DS_W, HDR_H, C_DS, BG_DS, "solid")
d.text(COL1_X + 10, y + 10, "DeepSpeed stock  (slow path)", FS_TITLE - 4, C_DS)
d.rect(COL2_X, y, COL_TU_W, HDR_H, C_TU, BG_TU, "solid")
d.text(COL2_X + 10, y + 10, "Tutel-patched  (fast path)", FS_TITLE - 4, C_TU)
y += HDR_H + ROW_GAP

# ━━ STEP 1: Routing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_label(COL0_X, y, COL_LABEL_W, ROW_H, 1, "Routing", "gate → topK")
cell(COL1_X, y, COL_DS_W, ROW_H, C_DS, BG_DS, "solid",
     "gate.forward → top2gating (Mixtral monkey-patch)",
     [
         "softmax(logits) → topK(2) → renormalize",
         "→ dense combine_weights [S, E, C]  (fp32)",
         "→ dense dispatch_mask  [S, E, C]  (fp32)",
         "via one_hot + cumsum for each of E experts",
         "Materializes ~32M fp32 cells in HBM every step",
     ])
cell(COL2_X, y, COL_TU_W, ROW_H, C_TU, BG_TU, "solid",
     "extract_critical(scores, top_k=2, capacity_factor, group)",
     [
         "softmax(logits) → topK(2) → renormalize  (same math)",
         "→ sparse (indices_, locations_, gates_)  len = S·K",
         "NO dense [S,E,C] mask materialized — 4000× fewer elements",
         "Supports arbitrary top_k (DS flag silently disables for k≠1)",
         "Returns l_aux load-balance loss identical to DS",
     ])
y += ROW_H + ROW_GAP

# ━━ STEP 2: Dispatch / Encode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_label(COL0_X, y, COL_LABEL_W, ROW_H, 2, "Dispatch", "pack → [E,C,M]")
cell(COL1_X, y, COL_DS_W, ROW_H, C_DS, BG_DS, "solid",
     'torch.einsum("sec,sm→ecm", dispatch_mask.float(), x)',
     [
         "Tensor shapes: [S,E,C] × [S,M] → [E,C,M]",
         "S=4096, E=8, C=1024, M=4096",
         "Work = S·E·C·M ≈ 137 G element-multiplies",
         "Non-zero fraction = K/E·C = 2/8192 ≈ 0.02%",
         "→ 99.98% of multiplies are ×0  (wasted work)",
     ])
cell(COL2_X, y, COL_TU_W, ROW_H, C_TU, BG_TU, "solid",
     "TutelMoeFastDispatcher.encode(x)   — fused CUDA kernel",
     [
         "Input: x [S,M] + sparse (indices_, locations_) len=S·K",
         "Work = S·K·M ≈ 33 M — exactly the non-zero assignments",
         "1 thread per (assignment, dim): coalesced gather from x,",
         "  coalesced write into [E, C, M] slot — no wasted work",
         "No fp32 intermediate, stays in bf16 throughout",
     ])
y += ROW_H + ROW_GAP

# ━━ STEP 3: AllToAll #1  (identical) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_label(COL0_X, y, COL_LABEL_W, ROW_H, 3, "AllToAll #1", "scatter to GPUs")
for cx, cw in [(COL1_X, COL_DS_W), (COL2_X, COL_TU_W)]:
    cell(cx, y, cw, ROW_H, C_SAME, BG_SAME, "solid",
         "_AllToAll.apply(self.ep_group, dispatched)   — UNCHANGED",
         [
             "dispatched reshaped to [ep_size, num_local_experts, C, M]",
             "dist.all_to_all_single: GPU i sends slice j to GPU j",
             "8×8 = 64 transfers per step  ·  each slice = C·M bf16 = 8 MB",
             "Same NCCL call in both paths — identical bytes on the wire",
             "AllToAll volume = ep_size · C · M · 2 bytes ≈ 64 MB / GPU",
         ])
# "SAME" badge
mid_x = (COL1_X + COL2_X + COL_TU_W) / 2
d.text(mid_x - 40, y + ROW_H // 2 - 8, "SAME IN BOTH", FS_NOTE + 2, C_SAME)
y += ROW_H + ROW_GAP

# ━━ STEP 4: Expert compute  (identical) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_label(COL0_X, y, COL_LABEL_W, ROW_H, 4, "Expert FFN", "SwiGLU compute")
for cx, cw in [(COL1_X, COL_DS_W), (COL2_X, COL_TU_W)]:
    cell(cx, y, cw, ROW_H, C_SAME, BG_SAME, "solid",
         "self.experts(dispatched)   — UNCHANGED",
         [
             "Same Experts module, same SwiGLU weights Wgu_j, Wd_j",
             "Input: [num_local_experts, C, M]  after AllToAll",
             "Wgu [2·EI, M] bmm → chunk → SiLU⊙ → Wd [M, EI] bmm",
             "Same cuBLAS grouped GEMM in both paths",
             "ZeRO-2 expert opt-state sharding also unchanged",
         ])
y += ROW_H + ROW_GAP

# ━━ STEP 5: AllToAll #2  (identical) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_label(COL0_X, y, COL_LABEL_W, ROW_H, 5, "AllToAll #2", "return to origin")
for cx, cw in [(COL1_X, COL_DS_W), (COL2_X, COL_TU_W)]:
    cell(cx, y, cw, ROW_H, C_SAME, BG_SAME, "solid",
         "_AllToAll.apply(self.ep_group, expert_output)   — UNCHANGED",
         [
             "expert_output reshaped to [ep_size · num_local_experts, C, M]",
             "Reverse exchange: GPU j returns e_j outputs to originating GPUs",
             "Same NCCL call in both paths — identical bytes on the wire",
             "Backward: autograd of _AllToAll emits symmetric AllToAll again",
             "AllToAll volume identical to step 3 ≈ 64 MB / GPU / direction",
         ])
y += ROW_H + ROW_GAP

# ━━ STEP 6: Combine / Decode ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_label(COL0_X, y, COL_LABEL_W, ROW_H, 6, "Combine", "weighted sum → [S,M]")
cell(COL1_X, y, COL_DS_W, ROW_H, C_DS, BG_DS, "solid",
     'torch.einsum("sec,ecm→sm", combine_weights, expert_out)',
     [
         "Tensor shapes: [S,E,C] × [E,C,M] → [S,M]",
         "Work = S·E·C·M ≈ 137 G element-multiplies  (same waste as step 2)",
         "combine_weights is fp32 → expert_out must be cast up for multiply",
         "Dynamo graph-break: aten._local_scalar_dense in one_hot path",
         "→ 99.98% of multiplies touch zero-weight slots (no-op)",
     ])
cell(COL2_X, y, COL_TU_W, ROW_H, C_TU, BG_TU, "solid",
     "TutelMoeFastDispatcher.decode(expert_out)   — fused CUDA kernel",
     [
         "Input: expert_out [E·C, M] + stored (indices_, gates_) from step 1",
         "Work = S·K·M ≈ 33 M — exactly the non-zero assignments",
         "1 thread per (assignment, dim): indexed gather from [E·C, M],",
         "  multiply by gate weight, atomic-add into out [S, M]",
         "Single kernel, no fp32 intermediate, no Dynamo graph-break",
     ])
y += ROW_H + ROW_GAP

# ━━ BOTTOM SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
y += 10
d.text(LEFT, y, "Summary", FS_TITLE, C_TXT); y += 34
for line in [
    "Steps 3, 4, 5  are IDENTICAL in both paths — same NCCL AllToAll bytes, same expert GEMM, same ZeRO-2 sharding.",
    "Steps 1, 2, 6  are replaced: sparse coordinate representation eliminates the dense [S,E,C] mask (32 M fp32 elements).",
    "Step 2 + 6 work: DS = S·E·C·M ≈ 137 G  vs  Tutel = S·K·M ≈ 33 M  →  ~4000× less work in dispatch/combine kernels.",
    "Wall-time speedup at production shape (S=4096, E=8, C=1024, M=4096, depth=4, 8×H100): 1052 ms → 562 ms  (1.87×).",
    "Why not use_tutel=True on DeepSpeedMoE? DS sets self.use_tutel = use_tutel and gate.k == 1 — silently off for top-2.",
]:
    d.text(LEFT + 10, y, line, FS_NOTE + 1, C_DIM); y += 22

out = os.path.join(os.path.dirname(__file__), "tutel_vs_ds.excalidraw")
d.write(out)
