"""Collect NVIDIA B200 device attributes via Nsight Compute.

Mirrors the "Session" page in the Nsight Compute UI.
Tries three approaches in order:
  1. NCU Python SDK  (ncu_report module bundled with the NCU install)
  2. ncu CLI import  (--page session or --print-details all, then text-parsed)
  3. torch + ctypes  (CUDA driver API fallback)

Saves results to reports/b200_device_attrs.json.
Usage: modal run src/modal/ncu_attr.py
"""
import sys, os
if os.path.isdir("/app"):
    sys.path.insert(0, "/app")

from src.modal.modal_utils import app, image

# Minimal script that launches at least one CUDA kernel so NCU has
# something to attach device-attribute metadata to.
_DUMMY_KERNEL = """\
import torch
x = torch.ones(1024, device="cuda", dtype=torch.float32)
y = x * 2.0          # triggers a kernel
torch.cuda.synchronize()
"""

# ── CUdevice_attribute enum ──────────────────────────────────────────────────
# Source: cuda.h / driver_types.h (CUDA 12.x).
# Keys are the snake_case names that the NCU Session page displays.
_CU_DEV_ATTRS: dict[str, int] = {
    "max_threads_per_block":                        1,
    "max_block_dim_x":                              2,
    "max_block_dim_y":                              3,
    "max_block_dim_z":                              4,
    "max_grid_dim_x":                               5,
    "max_grid_dim_y":                               6,
    "max_grid_dim_z":                               7,
    "max_shared_memory_per_block":                  8,
    "total_constant_memory":                        9,
    "warp_size":                                   10,
    "max_pitch":                                   11,
    "max_registers_per_block":                     12,
    "clock_rate":                                  13,
    "texture_alignment":                           14,
    "gpu_overlap":                                 15,
    "multiprocessor_count":                        16,
    "kernel_exec_timeout":                         17,
    "integrated":                                  18,
    "can_map_host_memory":                         19,
    "compute_mode":                                20,
    "maximum_texture1d_width":                     21,
    "maximum_texture2d_width":                     22,
    "maximum_texture2d_height":                    23,
    "maximum_texture3d_width":                     24,
    "maximum_texture3d_height":                    25,
    "maximum_texture3d_depth":                     26,
    "maximum_texture2d_layered_width":             27,
    "maximum_texture2d_layered_height":            28,
    "maximum_texture2d_layered_layers":            29,
    "surface_alignment":                           30,
    "concurrent_kernels":                          31,
    "ecc_enabled":                                 32,
    "pci_bus_id":                                  33,
    "pci_device_id":                               34,
    "tcc_driver":                                  35,
    "memory_clock_rate":                           36,
    "global_memory_bus_width":                     37,
    "l2_cache_size":                               38,
    "max_threads_per_multiprocessor":              39,
    "async_engine_count":                          40,
    "unified_addressing":                          41,
    "maximum_texture1d_layered_width":             42,
    "maximum_texture1d_layered_layers":            43,
    "maximum_texture2d_gather_width":              45,
    "maximum_texture2d_gather_height":             46,
    "maximum_texture3d_width_alternate":           47,
    "maximum_texture3d_height_alternate":          48,
    "maximum_texture3d_depth_alternate":           49,
    "pci_domain_id":                               50,
    "texture_pitch_alignment":                     51,
    "maximum_texturecubemap_width":                52,
    "maximum_texturecubemap_layered_width":        53,
    "maximum_texturecubemap_layered_layers":       54,
    "maximum_surface1d_width":                     55,
    "maximum_surface2d_width":                     56,
    "maximum_surface2d_height":                    57,
    "maximum_surface3d_width":                     58,
    "maximum_surface3d_height":                    59,
    "maximum_surface3d_depth":                     60,
    "maximum_surface1d_layered_width":             61,
    "maximum_surface1d_layered_layers":            62,
    "maximum_surface2d_layered_width":             63,
    "maximum_surface2d_layered_height":            64,
    "maximum_surface2d_layered_layers":            65,
    "maximum_surfacecubemap_width":                66,
    "maximum_surfacecubemap_layered_width":        67,
    "maximum_surfacecubemap_layered_layers":       68,
    "maximum_texture1d_linear_width":              69,
    "maximum_texture2d_linear_width":              70,
    "maximum_texture2d_linear_height":             71,
    "maximum_texture2d_linear_pitch":              72,
    "maximum_texture2d_mipmapped_width":           73,
    "maximum_texture2d_mipmapped_height":          74,
    "compute_capability_major":                    75,
    "compute_capability_minor":                    76,
    "maximum_texture1d_mipmapped_width":           77,
    "stream_priorities_supported":                 78,
    "global_l1_cache_supported":                   79,
    "local_l1_cache_supported":                    80,
    "max_shared_memory_per_multiprocessor":        81,
    "max_registers_per_multiprocessor":            82,
    "managed_memory":                              83,
    "is_multi_gpu_board":                          84,
    "multi_gpu_board_group_id":                    85,
    "host_native_atomic_supported":                86,
    "single_to_double_precision_perf_ratio":       87,
    "pageable_memory_access":                      88,
    "concurrent_managed_access":                   89,
    "compute_preemption_supported":                90,
    "can_use_host_pointer_for_registered_mem":     91,
    "cooperative_launch":                          95,
    "cooperative_multi_device_launch":             96,
    "max_shared_memory_per_block_optin":           97,
    "can_flush_remote_writes":                     98,
    "host_register_supported":                     99,
    "pageable_memory_access_uses_host_page_tables":100,
    "direct_managed_mem_access_from_host":         101,
    "virtual_memory_management_supported":         102,
    "handle_type_posix_file_descriptor_supported": 103,
    "handle_type_win32_handle_supported":          104,
    "handle_type_win32_kmt_handle_supported":      105,
    "max_blocks_per_multiprocessor":               106,
    "accessor_memory_pool_supported":              107,
    "sparse_cuda_array_supported":                 108,
    "read_only_host_register_supported":           109,
    "timeline_semaphore_interop_supported":        110,
    "memory_pools_supported":                      111,
    "gpu_direct_rdma_supported":                   112,
    "gpu_direct_rdma_flush_writes_options":        113,
    "gpu_direct_rdma_writes_ordering":             114,
    "mempool_supported_handle_types":              115,
    "cluster_launch":                              119,
    "deferred_mapping_cuda_array_supported":       121,
    "can_use_64_bit_stream_mem_ops":               122,
    "can_use_64_bit_stream_mem_ops_v1":            123,
    "can_use_stream_wait_value_nor":               124,
    "dma_buf_supported":                           125,
    "ipc_event_supported":                         126,
    "mem_sync_domain_count":                       127,
    "tensor_map_access_supported":                 128,
    "handle_type_fabric_supported":                132,
    "unified_function_pointers":                   134,
    "numa_config":                                 135,
    "numa_id":                                     136,
    "multicast_supported":                         137,
    "mps_enabled":                                 138,
    "host_numa_id":                                139,
    "gpu_pci_device_id":                           141,
    "gpu_pci_subsystem_id":                        142,
    "can_use_stream_mem_ops_v1":                   143,
    "can_use_stream_wait_value_nor_v1":            144,
}


# ── GPC count by compute capability ─────────────────────────────────────────────
# GPCs are a microarchitecture property, not a per-SKU property.  All SKUs
# within the same architecture share the same GPC count on the full die;
# harvested parts have the same number of GPCs but some TPCs disabled.
# Source: NVIDIA architecture whitepapers.
_GPC_BY_CC: dict[tuple, int] = {
    (7, 0): 6,   # Volta   (GV100)
    (7, 5): 6,   # Turing  (TU102)
    (8, 0): 8,   # Ampere  (GA100)
    (8, 6): 7,   # Ampere  (GA102 – RTX 3090/A6000)
    (8, 9): 7,   # Ada     (AD102 – RTX 4090/L40)
    (9, 0): 8,   # Hopper  (GH100)
    (10, 0): 8,  # Blackwell (GB100/GB200)
}


# ── Peak FLOPS table ────────────────────────────────────────────────────────
# Key: (cc_major, cc_minor)  →  (cores_per_sm, tc_per_sm, bf16_flops_per_tc_per_cycle)
# bf16_flops_per_tc_per_cycle counts fused-multiply-add (2 FLOP) operations.
# Sources: NVIDIA architecture whitepapers + dquery_peak.cu analysis.
_FLOPS_TABLE: dict[tuple, tuple] = {
    (7, 0): (64,  8,  128),   # V100 (GV100)
    (7, 5): (64,  8,  128),   # T4   (TU102)
    (8, 0): (64,  4,  512),   # A100 (GA100)
    (8, 6): (128, 4,  256),   # RTX 3090 / A6000 (GA102)
    (8, 9): (128, 4,  256),   # RTX 4090 / L40   (Ada Lovelace)
    (9, 0): (128, 4, 1024),   # H100 (GH100)
    (10, 0): (128, 4, 2048),  # B200 (GB100)
}


@app.function(image=image, gpu="B200", timeout=300)
def collect_attrs() -> dict:
    import subprocess, glob, json, re, torch, ctypes, io, csv as csv_mod

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    ncu_ver_dir = os.path.dirname(ncu)          # e.g. /opt/nvidia/nsight-compute/2026.1.0
    print(f"[ncu] {ncu}")

    # ── Step 1: generate a minimal NCU report ────────────────────────────────
    target = "/tmp/_ncu_attr_target.py"
    with open(target, "w") as f:
        f.write(_DUMMY_KERNEL)

    r = subprocess.run(
        [ncu, "-f", "--export", "/tmp/ncu_attr",
         "--set", "default",          # captures device info
         "--target-processes", "all",
         "python", target],
        capture_output=True, text=True, timeout=120,
    )
    print(f"[ncu capture] exit={r.returncode}")
    if r.stdout:
        print(r.stdout[-1000:])
    if r.returncode != 0 and r.stderr:
        print(f"[stderr] {r.stderr[-500:]}")

    # ── Step 2a: try NCU Python SDK (ncu_report) ─────────────────────────────
    ncu_sdk_attrs: dict = {}
    try:
        sdk_lib = sorted(glob.glob(f"{ncu_ver_dir}/python/lib"))
        if sdk_lib:
            sys.path.insert(0, sdk_lib[0])
        import ncu_report                                          # type: ignore
        ctx = ncu_report.load_report("/tmp/ncu_attr.ncu-rep")
        for ri in range(ctx.num_ranges()):
            rng = ctx[ri]
            for ai in range(rng.num_actions()):
                action = rng[ai]
                dev = action.device_attributes()
                for attr in dev:
                    k = str(attr.name()).lower().replace(" ", "_")
                    v = attr.value()
                    try:
                        ncu_sdk_attrs[k] = int(v)
                    except (TypeError, ValueError):
                        ncu_sdk_attrs[k] = str(v)
        print(f"[ncu_report SDK] {len(ncu_sdk_attrs)} attributes")
    except Exception as e:
        print(f"[ncu_report SDK] unavailable: {e}")

    # ── Step 2b: NCU CLI import --print-details all (text parse fallback) ────
    ncu_cli_attrs: dict = {}
    if not ncu_sdk_attrs:
        det = subprocess.run(
            [ncu, "--import", "/tmp/ncu_attr.ncu-rep", "--print-details", "all"],
            capture_output=True, text=True, timeout=60,
        )
        text = det.stdout
        print(f"[ncu --print-details] {len(text)} chars, snippet:\n{text[:1500]}")
        # Parse "  attribute_name     value" blocks after a "Device Attr…" header
        in_block = False
        for line in text.splitlines():
            low = line.lower()
            if "device attr" in low or "device properties" in low:
                in_block = True
                continue
            if in_block:
                stripped = line.strip()
                if not stripped or re.match(r"={3,}", stripped):
                    if ncu_cli_attrs:
                        break          # end of block
                    continue
                # Two or more spaces separating key and value
                parts = re.split(r" {2,}", stripped, maxsplit=1)
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    try:
                        ncu_cli_attrs[k] = int(v)
                    except ValueError:
                        try:
                            ncu_cli_attrs[k] = float(v)
                        except ValueError:
                            ncu_cli_attrs[k] = v
        print(f"[ncu CLI parse] {len(ncu_cli_attrs)} attributes")

    # ── Step 3: CUDA driver API via ctypes ───────────────────────────────────
    cuda_attrs: dict = {}
    try:
        libcuda = ctypes.cdll.LoadLibrary("libcuda.so.1")
        libcuda.cuInit(0)
        dev = ctypes.c_int(0)
        libcuda.cuDeviceGet(ctypes.byref(dev), 0)
        val = ctypes.c_int(0)
        for name, enum_id in _CU_DEV_ATTRS.items():
            ret = libcuda.cuDeviceGetAttribute(ctypes.byref(val), enum_id, dev)
            if ret == 0:               # CUDA_SUCCESS
                cuda_attrs[name] = val.value
        print(f"[ctypes CU driver] {len(cuda_attrs)} attributes")
    except Exception as e:
        print(f"[ctypes CU driver] failed: {e}")

    # ── Step 4: Derived topology ─────────────────────────────────────────────
    # sms_per_tpc = 2 is an architectural constant for all NVIDIA GPUs since Kepler.
    sm_count    = cuda_attrs["multiprocessor_count"]
    cc_major    = cuda_attrs["compute_capability_major"]
    cc_minor    = cuda_attrs["compute_capability_minor"]
    sms_per_tpc = 2
    total_tpcs  = sm_count // sms_per_tpc   # exact: mirrors NCU limits_num_tpcs

    gpcs = _GPC_BY_CC.get((cc_major, cc_minor))
    topo_source = "derived" if gpcs else "derived_no_gpc"
    sms_per_gpc   = (sm_count  // gpcs) if gpcs else None
    tpcs_per_gpc  = (total_tpcs // gpcs) if gpcs else None

    # Cluster size limits (Thread Block Clusters, Hopper+).
    # Portable:     max cluster size usable on any CC≥9 GPU without special opt-in.
    # Non-portable: larger limit, requires cudaFuncAttributeNonPortableClusterSizeAllowed.
    #               Only CC≥10 (Blackwell) supports 16; Hopper max is 8 portable.
    # NOTE: larger clusters reduce occupancy — cudaOccupancyMaxActiveClusters should
    #       be used per-kernel to verify.  Analytical bound:
    #         max_active_clusters_per_gpc = floor(smem_per_gpc / (cluster_size * smem_per_block))
    if cc_major >= 10:
        max_blocks_per_cluster_portable     = 8
        max_blocks_per_cluster_nonportable  = 16  # B200: opt-in via NonPortableClusterSizeAllowed
    elif cc_major >= 9:
        max_blocks_per_cluster_portable     = 8
        max_blocks_per_cluster_nonportable  = 8   # Hopper: 8 is already the max
    else:
        max_blocks_per_cluster_portable     = 0
        max_blocks_per_cluster_nonportable  = 0

    # Smem-based cluster occupancy (analytical, assumes 1 block/SM worst-case smem usage).
    # smem_per_gpc = sms_per_gpc * max_shared_memory_per_multiprocessor (bytes)
    max_smem_per_sm_bytes = cuda_attrs.get("max_shared_memory_per_block_optin", 0)
    smem_per_gpc_kb = (sms_per_gpc * max_smem_per_sm_bytes // 1024) if sms_per_gpc else None

    topology = {
        "source":                               topo_source,
        "compute_capability":                   f"{cc_major}.{cc_minor}",
        "multiprocessor_count":                 sm_count,
        "sms_per_tpc":                          sms_per_tpc,
        "total_tpcs":                           total_tpcs,
        "gpcs":                                 gpcs,
        "tpcs_per_gpc":                         tpcs_per_gpc,
        "sms_per_gpc":                          sms_per_gpc,
        "smem_per_gpc_kb":                      smem_per_gpc_kb,
        "max_blocks_per_cluster_portable":      max_blocks_per_cluster_portable,
        "max_blocks_per_cluster_nonportable":   max_blocks_per_cluster_nonportable,
    }
    print("[topology]", json.dumps(topology, indent=2))

    # ── Step 5: Peak FLOPS ───────────────────────────────────────────────────
    flops_entry = _FLOPS_TABLE.get((cc_major, cc_minor))
    if flops_entry:
        cores_per_sm, tc_per_sm, bf16_per_tc = flops_entry
        flops_source = "flops_table"
    else:
        cores_per_sm, tc_per_sm, bf16_per_tc = 128, 4, 2048   # Blackwell default
        flops_source = "default_blackwell"

    boost_hz        = cuda_attrs["clock_rate"] * 1e3
    mem_clock_hz    = cuda_attrs["memory_clock_rate"] * 1e3
    bus_width_bits  = cuda_attrs["global_memory_bus_width"]

    total_cuda_cores  = cores_per_sm * sm_count
    total_tensor_cores = tc_per_sm * sm_count

    fp32_peak_tflops  = total_cuda_cores * 2.0 * boost_hz / 1e12
    bf16_peak_tflops  = total_tensor_cores * bf16_per_tc * boost_hz / 1e12
    # Memory BW: mem_clock * 2 (DDR) * bus_width_bits / 8
    mem_bw_gbs        = (mem_clock_hz * 2.0 * bus_width_bits / 8) / 1e9

    fp32_ridge = fp32_peak_tflops * 1e12 / (mem_bw_gbs * 1e9)
    bf16_ridge = bf16_peak_tflops * 1e12 / (mem_bw_gbs * 1e9)

    peak_flops = {
        "source":                       flops_source,
        "cores_per_sm":                 cores_per_sm,
        "tc_per_sm":                    tc_per_sm,
        "bf16_flops_per_tc_per_cycle":  bf16_per_tc,
        "total_cuda_cores":             total_cuda_cores,
        "total_tensor_cores":           total_tensor_cores,
        "boost_clock_mhz":              round(cuda_attrs["clock_rate"] / 1e3, 1),
        "fp32_peak_tflops":             round(fp32_peak_tflops, 2),
        "bf16_tc_peak_tflops":          round(bf16_peak_tflops, 2),
        "mem_bw_gbs":                   round(mem_bw_gbs, 1),
        "fp32_ridge_flop_per_byte":     round(fp32_ridge, 1),
        "bf16_ridge_flop_per_byte":     round(bf16_ridge, 1),
    }
    print("[peak_flops]", json.dumps(peak_flops, indent=2))

    # ── Assemble result ───────────────────────────────────────────────────────
    # Priority: SDK attrs > CLI attrs > ctypes attrs (most authoritative first)
    ncu_section = ncu_sdk_attrs or ncu_cli_attrs

    result = {
        "device_name":              torch.cuda.get_device_name(0),
        "topology":                 topology,
        "peak_flops":               peak_flops,
        "ncu_device_attributes":    ncu_section,
        "cuda_driver_attributes":   cuda_attrs,
    }
    print(json.dumps(result, indent=2)[:3000])
    return result


@app.local_entrypoint()
def main():
    import json
    attrs = collect_attrs.remote()
    os.makedirs("reports", exist_ok=True)
    gpu_slug = attrs["device_name"].lower().replace(" ", "_")
    out_path = f"reports/{gpu_slug}_device_attrs.json"
    with open(out_path, "w") as f:
        json.dump(attrs, f, indent=2)
    print(f"\nSaved → {out_path}")
    t = attrs.get('topology', {})
    print(f"  topology:")
    print(f"    GPCs                  : {t.get('gpcs')}  (source: {t.get('source')})")
    print(f"    TPCs/GPC              : {t.get('tpcs_per_gpc')}")
    print(f"    SMs/TPC               : {t.get('sms_per_tpc')}")
    print(f"    total TPCs            : {t.get('total_tpcs')}")
    print(f"    SMs/GPC               : {t.get('sms_per_gpc')}")
    print(f"    max_blocks_per_cluster (portable)    : {t.get('max_blocks_per_cluster_portable')}")
    print(f"    max_blocks_per_cluster (non-portable): {t.get('max_blocks_per_cluster_nonportable')}")
    print(f"    smem_per_gpc_kb                      : {t.get('smem_per_gpc_kb')} KB")
    # print(f"  ncu_device_attributes   : {len(attrs.get('ncu_device_attributes', {}))} entries")
    # print(f"  cuda_driver_attributes  : {len(attrs.get('cuda_driver_attributes', {}))} entries")
    pf = attrs.get('peak_flops', {})
    print(f"  peak_flops:")
    print(f"    FP32 peak             : {pf.get('fp32_peak_tflops')} TFLOPS")
    print(f"    BF16 TC peak          : {pf.get('bf16_tc_peak_tflops')} TFLOPS")
    print(f"    Memory BW             : {pf.get('mem_bw_gbs')} GB/s")
    print(f"    FP32 roofline ridge   : {pf.get('fp32_ridge_flop_per_byte')} FLOP/byte")
    print(f"    BF16 roofline ridge   : {pf.get('bf16_ridge_flop_per_byte')} FLOP/byte")
