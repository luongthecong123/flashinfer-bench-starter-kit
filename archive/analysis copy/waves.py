WORKLOAD_INFO = [
    ("0c23b10c", 1, [2]),
    ("9d4a5f21", 2, [18, 11]),
    ("b7668cfd", 2, [33, 52]),
    ("0a63b87b", 2, [63, 9]),
    ("05f6de65", 2, [6, 337]),
    ("fc85411e", 2, [17, 13]),
    ("e6b849f2", 2, [92, 48]),
    ("9f3f891b", 2, [288, 4]),
    ("f77df5ce", 2, [18, 19]),
    ("385742b2", 8, [92, 48, 1044, 14, 411, 30, 16, 8]),
    ("4c46a94b", 8, [18, 19, 1002, 31, 11, 316, 24, 2]),
    ("38389961", 7, [33, 52, 72, 17, 18, 401, 1089]),
    ("02d6ae9c", 8, [63, 9, 2048, 212, 11, 25, 6, 50]),
    ("ddfa9e34", 6, [6, 9, 9, 14, 1639, 71]),
    ("78b2e11c", 8, [18, 11, 2048, 20, 25, 45, 135, 326]),
    ("68d6817d", 6, [19, 20, 32, 12, 25, 3]),
    ("564007ac", 8, [288, 4, 1884, 21, 136, 2048, 42, 335]),
    ("ae4219a9", 7, [19, 12, 2048, 21, 26, 46, 136]),
    ("232ed014", 8, [35, 54, 74, 19, 20, 403, 1091, 1]),
    ("7a389715", 8, [8, 11, 11, 16, 1641, 73, 1, 1]),
    ("5096e459", 8, [17, 13, 1887, 16, 180, 1986, 413, 1]),
    ("d57eb9e1", 6, [143, 139, 2013, 142, 306, 539]),
    ("2207f0fd", 7, [415, 131, 2011, 148, 263, 169, 462]),
]

NUM_SM = 148
H = 16
# NUM_SPLIT = 8
TOPK = 2048
DIM_SPLIT = 512
BLOCK_SIZE = 1024
WORKLOAD_IDX = 20 - 1

MAX_SMEM_PER_SM = 228
SMEM_PER_BLOCK = 200
MAX_BLOCK_PER_SM = MAX_SMEM_PER_SM // SMEM_PER_BLOCK # Limited by smem usage

for idx, workload_idx in enumerate(range(len(WORKLOAD_INFO))):
    # print(f"Workload {workload_idx}: {WORKLOAD_INFO[workload_idx][2]} nonzeros")

    print(idx + 1,  WORKLOAD_INFO[workload_idx][2])
    work_list = WORKLOAD_INFO[workload_idx][2]
    num_blocks = []

    for work_idx in range(len(work_list)):
        n = (work_list[work_idx] + DIM_SPLIT - 1) // DIM_SPLIT
        # print(f"Workload {work_idx}: {work_list[work_idx]} nonzeros -> {n}x{H} blocks")
        num_blocks.append(n)

    total_blocks = sum(num_blocks) * H
    print(f"Total blocks: {total_blocks}")

    blocks_per_sm = min((2048 // BLOCK_SIZE), MAX_BLOCK_PER_SM) # One sm can handle blocks_per_sm blocks
    num_sm_required = total_blocks // blocks_per_sm
    num_waves = num_sm_required / NUM_SM
    print(f"Blocks per SM: {blocks_per_sm}, SMs required: {num_sm_required:.2f}, Waves: {num_waves:.2f}")

# ── DIM_SPLIT sweep for load-balance analysis ─────────────────────────────────
print("\n" + "="*70)
print("DIM_SPLIT sweep — KV split granularity vs. load balance")
print("="*70)
print("Each block processes up to DIM_SPLIT valid KV indices.")
print("Final reduction across splits done via DSMEM.\n")
print(f"{'DIM_SPLIT':>10}  {'WL':>4}  {'valid_counts':<30}  {'blocks_per_token':<35}  {'total_blks':>10}  {'waves':>6}  {'imbalance':>10}")
print("-"*130)

DIM_SPLIT_SWEEP = [2048, 1024, 512, 256, 128, 64]

for dim_split in DIM_SPLIT_SWEEP:
    for idx, (wl_id, num_tokens, work_list) in enumerate(WORKLOAD_INFO):
        num_blocks = [(v + dim_split - 1) // dim_split for v in work_list]
        total_blocks = sum(num_blocks) * H
        blocks_per_sm = min((2048 // BLOCK_SIZE), MAX_BLOCK_PER_SM)
        num_sm_required = total_blocks // blocks_per_sm
        num_waves = num_sm_required / NUM_SM
        max_splits = max(num_blocks)
        min_splits = min(num_blocks)
        imbalance = max_splits / (sum(num_blocks) / len(num_blocks))
        print(f"{dim_split:>10}  {idx+1:>4}  {str(work_list):<30}  {str(num_blocks):<35}  {total_blocks:>10}  {num_waves:>6.2f}  {imbalance:>10.2f}x")
    print()

# ── Focus on the most extreme workload ────────────────────────────────────────
print("="*70)
print("Focus: workload 20 — 7a389715  [8, 11, 11, 16, 1641, 73, 1, 1]")
print("  Token 4 has 1641 valid KV vs. min=1, max=1641 (1641x imbalance)")
print("="*70)
EXTREME_WL = [8, 11, 11, 16, 1641, 73, 1, 1]
print(f"\n{'DIM_SPLIT':>10}  {'splits_per_token':<35}  {'total_blks':>10}  {'waves':>6}  {'max_work_per_blk':>18}  {'imbalance':>10}")
print("-"*100)
for dim_split in DIM_SPLIT_SWEEP:
    splits = [(v + dim_split - 1) // dim_split for v in EXTREME_WL]
    total_blocks = sum(splits) * H
    blocks_per_sm = min((2048 // BLOCK_SIZE), MAX_BLOCK_PER_SM)
    num_waves = (total_blocks // blocks_per_sm) / NUM_SM
    max_work = dim_split  # upper bound on KV entries processed per block
    imbalance = max(splits) / (sum(splits) / len(splits))
    print(f"{dim_split:>10}  {str(splits):<35}  {total_blocks:>10}  {num_waves:>6.2f}  {max_work:>18}  {imbalance:>10.2f}x")

# ── Cluster occupancy: how cluster_size reduces active blocks ─────────────────
# With TBC clusters, all `cluster_size` blocks must be co-scheduled on the same GPC.
# The smem budget of the entire GPC caps the number of simultaneously active clusters:
#   max_active_clusters_per_gpc = floor(smem_per_gpc_kb / (cluster_size * SMEM_PER_BLOCK))
#   total_active_clusters       = num_gpcs * max_active_clusters_per_gpc
#   total_active_blocks         = total_active_clusters * cluster_size
# Portable cluster limit: 8  (CC >= 9.0)
# Non-portable limit:     16 (CC >= 10.0, requires cudaFuncAttributeNonPortableClusterSizeAllowed)
print("\n" + "="*70)
print("Cluster occupancy: smem-limited active clusters per GPC")
print("       (B200: 8 GPCs, 18 SMs/GPC, 228 KB smem/SM)")
print("="*70)
NUM_GPCS        = 8
SMS_PER_GPC     = NUM_SM // NUM_GPCS          # 148 // 8 = 18 (rounded)
SMEM_PER_GPC_KB = SMS_PER_GPC * MAX_SMEM_PER_SM  # 18 * 228 = 4104 KB
print(f"smem_per_gpc = {SMS_PER_GPC} SMs × {MAX_SMEM_PER_SM} KB = {SMEM_PER_GPC_KB} KB\n")

print(f"{'cluster_size':>14}  {'portable':>16}  {'max_clusters/GPC':>18}  {'total_active_clusters':>22}  {'total_active_blocks':>20}  {'gpu_occupancy%':>15}")
print("-"*112)
for cluster_size in [1, 2, 4, 8, 16]:
    portable = "yes" if cluster_size <= 8 else "no (B200-only)"
    smem_needed_kb = cluster_size * SMEM_PER_BLOCK     # KB needed per cluster on GPC
    max_clusters_per_gpc = SMEM_PER_GPC_KB // smem_needed_kb
    total_active_clusters = NUM_GPCS * max_clusters_per_gpc
    # Active blocks cannot exceed SM count (each block occupies exactly 1 SM)
    total_active_blocks   = min(total_active_clusters * cluster_size, NUM_SM)
    occupancy_pct = 100.0 * total_active_blocks / NUM_SM
    print(f"{cluster_size:>14}  {portable:>16}  {max_clusters_per_gpc:>18}  {total_active_clusters:>22}  {total_active_blocks:>20}  {occupancy_pct:>14.1f}%")
print(f"\nNote: cluster_size=1 = no cluster (baseline). SMEM_PER_BLOCK={SMEM_PER_BLOCK} KB assumed.")
