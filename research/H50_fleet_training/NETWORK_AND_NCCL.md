# H50: fleet network and NCCL measurements (h50-fleet, 2026-11-23, EXPLORATORY)

Measured by h50-fleet between 06:00Z and 06:50Z, before any H50 topology or DiLoCo run. The source is h50-fleet's posts in `AGENT_CHAT.md` (06:12:40Z, 06:23:57Z, 06:28:27Z, 06:50:09Z and 06:53:44Z), with the raw rows in `results/`. Raw originals, tools and their sha256 are listed in [MANIFEST](MANIFEST). The DiLoCo pre-registration is in [DILOCO.md](DILOCO.md), and its run settings are in `configs/`.

## Network (`results/network_map.json`, `cloud_*.jsonl`, `lan_tcp.jsonl`, `wan_tcp.jsonl`)
- **Cloud A100 island** (private network, MTU 9000):
  - Ping RTT 0.20–0.42 ms average.
  - TCP throughput: 16–23 Gbit/s on one stream, 79–109 Gbit/s on eight.
  - RoCE v2 `ib_write_bw` reaches 111 Gbit/s; `ib_send_lat` is 1.37 µs.
  - The 30-vCPU hosts (xbabe8, xbabe9) retransmit under 4–8 streams.
- **Site ↔ cloud:** only port 22 is open inbound. RTT is 28–29 ms. Throughput is ~115 Mbit/s cloud → site and ~245 Mbit/s site → cloud, so the site link is the cap.
- **On-prem LAN:** 1 GbE.
- **Name resolution:** cloud hosts resolve no `xbabe*` name, and on-prem `/etc/hosts` entries are stale. Dial IP addresses or use one shared hosts file.

## NCCL (`results/nccl_local_*.jsonl`, `results/xhost_*.jsonl`)
- **Intra-node, xbabe5's 2× A100 PCIe:** the VM disables P2P, so NCCL falls back to shared memory.
  - Default all-reduce bus bandwidth is 3.0 GB/s.
  - `NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3` raises it to **16.9 GB/s**.
- **Cross-host, 2 ranks (xbabe5 GPU0 + xbabe9):**
  - Torch's NCCL 2.26.2 loads the system RDMA plugin (`NET/IBext_v10`), so traffic runs over RoCE even with `NCCL_IB_DISABLE=1`. A1 therefore trained over RDMA, not sockets.
  - Default: 3.4 GB/s, because GPUDirect RDMA is off (GPU ↔ NIC is PHB).
  - `NCCL_NET_GDR_LEVEL=SYS`: **6.3 GB/s (1.8×)**.
  - Socket transport (`NCCL_NET=Socket`): 1.8–2.0 GB/s.
- **Flagged rows:** the sweep in `xhost_064834b.jsonl` overlapped the first 28 s of H48b seed 1234 on xbabe5 GPU0 (h50-fleet, 06:50:09Z). The clean sweep is `xhost_064708.jsonl`.
- **Tuned FP32 profile proposed for registration:**
  - intra-node: the SHM-memcpy pair above;
  - cross-host: `NCCL_NET_GDR_LEVEL=SYS`;
  - kept: `NCCL_SOCKET_IFNAME=eno1`, and `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` in every run log.
  - **Still open:** an exact-sum correctness check of GDR, since bandwidth alone proves nothing.
  - codex-lead (06:39:50Z) ruled that the production scaling verdict uses the tuned FP32 arm, with A1's environment kept as a separately identified comparator.

## DDP step probe (synthetic batches, real S model, 56.7 M parameters, xbabe5 2× A100)
At the 750 s-global / 2-GPU shape (375 s per rank), comparing step time with and without gradient sync:

| comm | NCCL env | step (sync) | step (no sync) | exposed comm |
|---|---|---|---|---|
| FP32 | default (A1's) | 177.3 ms | 112.9 ms | 64.3 ms = 36.3% |
| FP32 | SHM memcpy pair | 119.8 ms | 113.4 ms | 6.4 ms = 5.3% |
| BF16 hook | default | 149.1 ms | 113.1 ms | 35.9 ms = 24.1% |

- At A1's per-rank shape (1,037 s, 83 utterances) in FP32, exposed comm is 54.5 ms (16.5%) with A1's environment and 4.2 ms (1.5%) with the SHM-memcpy pair.
- BF16 communication adds nothing once the transport is fixed: 7.0 ms vs 6.4 ms at the 375 s shape. That fails codex-lead's ≥ 10% bar on xbabe5.
- **A1's world-5 log, read with the harness:** 9,262 audio-h/h at 4 ranks, then 11,218 and 10,916 at 5 ranks. Throttle sleep was 2–3% of wall time, data wait 4–5%, backward + collective 52% of the step, GPU utilisation 76–79%. A1's ~15% loss was gradient communication and host stalls, not the throttle.

## Not yet measured
- ≥ 92% per-rank efficiency: six A100s are unavailable while the co-tenant holds xbabe7 / xbabe8.
- The topology ladder (T1 / T2L / T2X) runs after trainer-lane's H44 PR (#80).
- The DiLoCo acceptance tests A1–A5 wait for the hub fixes (#73).
