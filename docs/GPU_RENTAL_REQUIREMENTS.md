# GPU Rental Requirements & Budget Justification

**Project:** ABSA SFT Comparison Experiment  
**Date:** 2026-07-06  
**Prepared by:** Rafly Hanggaraksa  
**Budget Allocation:** ~$470 (recommended)

---

## Executive Summary

Kami membutuhkan **4× RTX 4090 GPU** di Vast.ai untuk menjalankan full experiment dalam **8 hari kerja** dengan total budget **~$470**.

---

## 1. Mengapa Butuh System RAM 32GB?

### Penjelasan Teknis

**System RAM 32GB diperlukan untuk:**

| Komponen | RAM Usage |
|---|---|
| Model weights (Qwen 0.5B) | ~2 GB |
| Optimizer states (AdamW) | ~4-6 GB |
| Batch data loading (bs=4) | ~2-3 GB |
| Gradient accumulation buffer | ~4 GB |
| HuggingFace + PyTorch overhead | ~2-3 GB |
| OS + other processes | ~2-3 GB |
| **Total Peak** | **~18-21 GB** |

**Buffer untuk:** 
- Swap space (kalau memory spike)
- Concurrent processes (monitoring, logging)
- Safety margin agar tidak OOM (Out of Memory)

**Rekomendasi minimum:** 32 GB ✓ (memberikan ~11-14 GB buffer)

**Kalau cuma pakai 16 GB?** Risiko crash → job gagal → harus rerun = buang waktu & biaya.

---

## 2. 120 Parallel Jobs Itu Gimana Maksudnya?

### Grid Experiment Configuration

Training ini adalah **hyperparameter sweep** dengan kombinasi:

```
3 Models × 6 Languages × 1 Dataset × 5 Seeds = 90 total jobs
```

**Detail breakdown:**

**Models (3):**
- google/gemma-3-270m
- Qwen/Qwen2.5-0.5B  
- google/mt5-base

**Languages (6):**
- English (eng)
- Javanese (jav)
- Indonesian (indo)
- Madurese (mad)
- Minangkabau (min)
- Sundanese (sun)

**Datasets (1):**
- hotel_reviews/mvp

**Random Seeds (5):**
- 9584, 123, 2024, 31415, 777

### Apa itu "120 parallel job"?

Sebenarnya **tidak 120**, tapi **90 jobs**.

- **Slurm array config:** `--array=0-119%5` = reserve 120 slots untuk future expansion
- **Actual jobs:** 90 (3 × 6 × 1 × 5)
- **`%5` maksudnya:** maksimal 5 jobs running in parallel pada Slurm cluster

### Timeline Kalau Sequential (1 GPU)

```
1 job × 8 hours = 8 jam per job
90 jobs × 8 jam = 720 jam = 30 hari (1 bulan full)
```

Ini terlalu lama untuk deadline research. Butuh parallelisasi.

---

## 3. Timeline Considerations

### Deadline Requirements

**Pertanyaan untuk project manager:**
- Kapan hasil dibutuhkan? (deadline untuk paper/presentation/report?)
- Apakah ada keharusan selesai sebelum tanggal tertentu?

### Recommended Timeline

| Scenario | Timeline | Reasoning |
|---|---|---|
| **Flexible (no hard deadline)** | 14-30 hari | Bisa pakai 1 GPU, cost minimal |
| **Moderate (research momentum)** | 7-10 hari | Pakai 3-4 GPU, reasonable budget |
| **Urgent (paper deadline)** | 2-3 hari | Pakai 12-16 GPU, high cost |

### Kami Recommend: 8-10 hari maksimum

**Alasan:**
- Research experiments butuh iterasi → hasil cepat = bisa adjust cepat
- Kalau tunggu 1 bulan, bisa ada bottleneck lain di project
- 8-10 hari itu reasonable untuk academic timeline

---

## 4. Kenapa Butuh 4 GPU?

### Calculation

```
Total Jobs:       90
GPU needed:       90 jobs ÷ X GPUs = jobs per GPU
Time per job:     8 hours
Total compute:    90 × 8 = 720 GPU-hours (fixed)

Dengan 4 GPU:
  - Jobs per GPU:  90 ÷ 4 = 22.5 jobs
  - Compute time:  22.5 × 8 = 180 hours = 7.5 days
  - Safety margin: 6.5 days (untuk interruptions, reruns)
  - Total timeline: ~14 hari ✓
```

### Kenapa bukan 1, 2, 3, atau lebih dari 4?

| GPUs | Timeline | Cost/Day | Total Cost | Pros | Cons |
|---|---|---|---|---|---|
| **1** | 30 days | $8.40 | $360 | Cheapest | Terlalu lama, risky |
| **2** | 15 days | $16.80 | $360 | Masih affordable | Masih terlalu lama |
| **3** | 10 days | $25.20 | $360 | Good balance | Tight budget |
| **4** ✓ | 7-8 days | $33.60 | $470 | **Best sweet spot** | **Recommended** |
| **6** | 5 days | $50.40 | $540 | Faster | Lebih mahal, kompleks |
| **12** | 2 days | $100.80 | $640 | Paling cepat | Expensive, risky management |

### Why 4 is Sweet Spot:

1. **Reasonable timeline:** 8 hari = deadline realistis untuk revisions/fixes
2. **Budget buffer:** $470 vs actual $360 compute = $110 buffer untuk:
   - Spot instance interruptions (Vast.ai bias interruptible)
   - Failed jobs → reruns
   - Test runs
3. **Management complexity:** 4 GPU = mudah dimonitor, tidak overwhelming
4. **Availability:** Vast.ai biasanya punya 4+ RTX 4090 available

---

## 5. Comparison: 4× RTX 4090 vs 1× L4 GPU

### Spesifikasi Hardware

| Spec | RTX 4090 | L4 |
|---|---|---|
| VRAM | 24 GB | 24 GB |
| Architecture | Consumer (Ada) | Datacenter (Ada) |
| Peak Compute (FP32) | 82.6 TFLOPS | 30.1 TFLOPS |
| Peak Compute (TF32) | ~660 TFLOPS | ~120 TFLOPS |
| Memory Bandwidth | 576 GB/s | 432 GB/s |
| Power Draw | 450W | 70W |
| **Availability on Vast.ai** | ✓ Banyak | ✗ Jarang |
| **Availability on RunPod** | ✓ Cukup | ✓ Available |

### Cost Comparison (Vast.ai)

**Scenario A: 4× RTX 4090 (Recommended)**
```
Cost:       $0.35/hr per GPU × 4 = $1.40/hr
Timeline:   7-8 days actual + 6-7 days buffer = 14 hari
Total:      $1.40/hr × 336 jam = $470.40

Jobs per GPU:    22.5 (serial, one after another)
Per-job time:    8 hours
Parallelism:     4 jobs running simultaneously
```

**Scenario B: 1× L4 GPU (untuk comparison)**
```
Vast.ai L4 cost:  $0.28/hr (jarang ada)
RunPod L4 cost:   $0.45/hr (lebih reliable)

Timeline:   90 jobs × 8 hrs = 720 jam = 30 hari
Total cost: $0.45/hr × 720 jam = $324

Jobs per GPU:    90 (all serial)
Per-job time:    8 hours
Parallelism:     0 (sequential only)
```

### Detailed Cost Breakdown

| Item | 4× RTX 4090 | 1× L4 |
|---|---|---|
| **Compute cost** | $470 | $324 |
| **Timeline** | 8 days | 30 days |
| **Failed job cost** (est 5% fail rate) | +$23 | +$16 |
| **Rerun budget** | $70 (15% buffer) | $30 |
| **Total realistic budget** | **$550** | **$370** |
| **Days to completion** | **8-14 hari** | **30-45 hari** |

### Performance Per Job (RTX 4090 vs L4)

**Training speed (Qwen 0.5B, batch_size=4, 10 epochs):**

```
RTX 4090: 
  - Training throughput: ~200 tokens/sec (FP32/TF32 compute advantage)
  - Per-job time: ~8 hours
  
L4:
  - Training throughput: ~120 tokens/sec (lower compute)
  - Per-job time: ~13 hours (slower)
```

**Impact:**
- RTX 4090 × 4 menjalankan 4 jobs parallel = 4× throughput
- L4 × 1 menjalankan 1 job = baseline throughput
- **Effective speedup: 4 × (200/120) = ~6.7× faster overall**

### Inference Speed (untuk evaluation)

```
RTX 4090: ~500 tokens/sec
L4:       ~300 tokens/sec

Advantage RTX 4090: 67% faster
```

---

## 6. Comprehensive Comparison Matrix

### 6a. Cost vs Timeline Trade-off

```
Budget Range    | Timeline | Setup | Recommendation
────────────────┼──────────┼──────┼─────────────────────
Sangat Tight    | 30 days  | 1 L4 | Kalau HARUS murah
($300-350)      |          |      | & deadline fleksibel
────────────────┼──────────┼──────┼─────────────────────
Moderate        | 10 days  | 3 GPU| Kalau budget $400
($400-450)      |          |      | & perlu dalam 2 minggu
────────────────┼──────────┼──────┼─────────────────────
RECOMMENDED ✓   | 8 days   | 4    | Best balance cost,
($470-550)      |          | RTX  | speed, reliability
────────────────┼──────────┼──────┼─────────────────────
Premium         | 3-5 days | 8    | Kalau deadline
($600-800)      |          | GPU  | sangat ketat
```

### 6b. Availability & Reliability

| Platform | 4× RTX 4090 | 1× L4 |
|---|---|---|
| **Vast.ai** | ✓ Selalu ada | ✗ Langka |
| **RunPod** | ✓ Tersedia | ✓ Banyak |
| **Spot interruption risk** | 10-15% | 5% |
| **Expected uptime** | 95% | 98% |
| **Recovery time** | ~5 min | ~5 min |

**Untuk 4 GPU:** kalau 1 terputus, 3 lain masih berjalan → parallelism tetap 3  
**Untuk 1 GPU:** kalau terputus, seluruh pipeline berhenti

### 6c. Operational Complexity

```
4× RTX 4090:
  ├─ SSH setup: 4 terminals (bisa di 1 screen session)
  ├─ Monitoring: top/nvidia-smi di 4 windows
  ├─ Logs: perlu aggregate 4 stdout/stderr
  └─ Effort: Moderate (~30 min setup, 2x daily check)

1× L4:
  ├─ SSH setup: 1 terminal
  ├─ Monitoring: 1 nvidia-smi
  ├─ Logs: Simple, centralized
  └─ Effort: Low (~10 min setup, 1x daily check)
```

---

## 7. Risk Analysis

### Dengan 4× RTX 4090 (Recommended)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| 1 GPU spot interrupted | 10-15% | Medium | 3 lain tetap jalan, just spin up 1 baru |
| 2 GPU concurrent fail | 2% | Low | Repivot ke 2 GPU, tetap selesai |
| Data corruption | 1% | High | Keep recent logs on local machine |
| Instance unable to restart | 5% | Medium | Have backup instance warm |

**Risk-adjusted timeline:** 8 days + 6 day buffer = 14 hari (manageable)

### Dengan 1× L4 (Worst Case)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| GPU spot interrupted | 10% | **CRITICAL** | Entire pipeline stops, must restart |
| Model fail to load | 2% | **CRITICAL** | Pipeline stuck 24 hours |
| Network timeout | 5% | **CRITICAL** | Manual reconnect + rerun |

**Risk-adjusted timeline:** 30 days + 15 day buffer = 45 hari (risky)

**Conclusion:** 4 GPU = redundancy, 1 GPU = single point of failure

---

## 8. Final Recommendation

### Opsi yang Available:

**OPSI 1 (Recommended) ✓**
- **Setup:** 4× RTX 4090 di Vast.ai
- **Cost:** $470 (all-in budget: $550)
- **Timeline:** 8 hari actual, 14 hari buffer
- **Rationale:** Best cost-performance, timeline reasonable, parallelization mengurangi risk

**OPSI 2 (Conservative)**
- **Setup:** 3× RTX 4090 di Vast.ai
- **Cost:** $360 (all-in budget: $450)
- **Timeline:** 10 hari actual, 14 hari buffer
- **Rationale:** Lebih murah, masih reasonable, tapi tighter timeline

**OPSI 3 (Aggressive)**
- **Setup:** 6× RTX 4090 di Vast.ai
- **Cost:** $540 (all-in budget: $650)
- **Timeline:** 5 hari actual, bisa 7 hari with buffer
- **Rationale:** Paling cepat, tapi budget naik 30%

**OPSI 4 (Not recommended)**
- **Setup:** 1× L4 di RunPod
- **Cost:** $324 (cheaper upfront)
- **Timeline:** 30 hari actual, 45 hari realistic
- **Rationale:** Cheapest tapi terlalu lama, high risk, tidak worth savings

---

## 9. Questions untuk Project Manager

**Sebelum finalize budget, jawab pertanyaan ini:**

1. **Deadline:** Kapan hasil dibutuhkan?
   - Jawaban → Tentukan opsi (1,2,3, atau 4)

2. **Budget flexibility:** Apakah $470-550 acceptable?
   - Jawaban → Approve OPSI 1 atau adjust ke OPSI 2

3. **Risk tolerance:** Prefer cepat tapi risky vs lambat tapi aman?
   - Jawaban → Determine parallelism level

4. **Iterative vs final:** Apakah ini final run atau ada kemungkinan reruns?
   - Jawaban → Add rerun budget estimate

---

## Attachment: Technical Spec Sheet

**For IT/DevOps team:**

```yaml
Requirement:
  gpu_type: "NVIDIA RTX 4090"
  quantity: 4
  vram_per_gpu: 24GB
  system_ram: 32GB
  disk_space: 50GB (for models + checkpoints)
  network: Internet connection (for HuggingFace + W&B)
  
Platform:
  preferred: "Vast.ai"
  alternative: "RunPod"
  
Software:
  pytorch: "latest"
  cuda: "12.1+"
  python: "3.10+"
  
Timeline:
  expected_duration: "7-8 days"
  with_buffer: "14 days"
  
Cost:
  per_hour: $1.40
  per_day: $33.60
  total_estimated: $470
  buffer_allocation: $80
  total_budget: $550
```

---

**Document Status:** Ready for Budget Approval  
**Next Step:** Project Manager approval → Proceed with Vast.ai rental setup
