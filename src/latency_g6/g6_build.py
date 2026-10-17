"""Build g6_latency.json + g6_latency_table.csv from the per-dataset profiler outputs (descriptive G6).
Post-processing only: end-to-end per-query latency = sum of per-query stage times + amortised per-snapshot cost.
Amortised snapshot cost (primary) = total snapshot time over all test timestamps / total test queries
  = (evolve_ms_mean + graph_ms_mean) * n_test_ts / n_test_queries   (ratio of sums; full test traffic).
The profiler's own field (mean over timestamps of per-timestamp ratios) is kept as 'amortised_ms_mean_of_ratios'."""
import json, sys, numpy as np, csv
D = sys.argv[1]; OUT = sys.argv[2]
Q = {"polecat": (532636, 304), "smallpedia": (163172, 17)}
proto = dict(sha256="0c0638d4002a9c1b08638d2c0df27d7ce26c7a3e15915d6a65a267c8c1612d75", registered_utc="2026-10-14T04:23:49Z")
res = dict(name="g6_latency", label="DESCRIPTIVE PROFILING (no decision rule; post-hoc nothing)", protocol=proto, units="ms per query unless stated; batched per-query latency = batch wall / batch size (protocol)", datasets={})
rows = []
for ds in ("polecat", "smallpedia"):
    J = json.load(open(f"{D}/g6_{ds}.json")); qn, nts = Q[ds]
    out = dict(n_sample=J["n_sample"], repeats=J["repeats"], warmup=1, chunk=J["chunk"], cpu_threads=J["threads"], base=dict(model="TiRGN", **J["tirgn"]),
               n_test_queries=qn, n_test_timestamps=nts, setup_s=J["setup_s"], wall_s=J["wall_s"], batch_size=J["batch_size"],
               agreement_vs_stored_lists=J["agreement_vs_stored_lists"], stages=J["stages"], batch_wall=J["batch_wall"], end_to_end={}, snapshot={},
               memory=dict(per_stage=J["memory"], torch_max_reserved_GB=J["gpu_max_reserved_GB"], host_rss_end_GB=J["rss_GB"], host_rss_probe=J["host_rss_probe"]),
               environment=J["environment"])
    for m in ("b1", "batched"):
        sn = J["snapshot"][m]; am_old = sn["amortised_ms_per_query_full_test_traffic"]
        am = (sn["evolve_ms_mean"] + sn["graph_ms_mean"]) * nts / qn
        out["snapshot"][m] = dict(evolve_ms_per_timestamp=sn["evolve_ms_mean"], graph_build_ms_per_timestamp=sn["graph_ms_mean"],
                                  amortised_ms_per_query=am, amortised_ms_mean_of_ratios=am_old)
        e = {}
        for arm, v in J["end_to_end"][m].items():
            sh = am - am_old
            mean = v["mean_ms"] + sh
            e[arm] = dict(p50_ms=v["p50_ms"] + sh, p95_ms=v["p95_ms"] + sh, mean_ms=mean, n=v["n"], throughput_qps=1e3 / mean,
                          throughput_qps_excl_snapshot=v["throughput_qps_excl_snapshot"], mean_ms_excl_snapshot=1e3 / v["throughput_qps_excl_snapshot"])
        for arm in ("HR", "HRCOV_PAT"):
            e[arm]["overhead_vs_base"] = e[arm]["mean_ms"] / e["base"]["mean_ms"] - 1.0
            e[arm]["overhead_vs_base_excl_snapshot"] = e[arm]["mean_ms_excl_snapshot"] / e["base"]["mean_ms_excl_snapshot"] - 1.0
        e["HRCOV_PAT"]["pattern_block_share_of_e2e"] = J["stages"][m]["S2_PAT"]["mean_ms"] / e["HRCOV_PAT"]["mean_ms"]
        out["end_to_end"][m] = e
        for arm, v in e.items():
            rows.append(dict(dataset=f"tkgl-{ds}", mode=m, row="end_to_end", name=arm, p50_ms=v["p50_ms"], p95_ms=v["p95_ms"], mean_ms=v["mean_ms"],
                             throughput_qps=v["throughput_qps"], overhead_vs_base=v.get("overhead_vs_base", 0.0), gpu_peak_GB="", host_hwm_delta_GB=""))
        rows.append(dict(dataset=f"tkgl-{ds}", mode=m, row="stage", name="S1_snapshot (per timestamp; amortised in end_to_end)", p50_ms="", p95_ms="",
                         mean_ms=sn["evolve_ms_mean"] + sn["graph_ms_mean"], throughput_qps="", overhead_vs_base="",
                         gpu_peak_GB=J["memory"]["S1_snapshot"]["gpu_peak_GB"], host_hwm_delta_GB=J["memory"]["S1_snapshot"]["host_hwm_delta_GB"]))
        for st, v in J["stages"][m].items():
            rows.append(dict(dataset=f"tkgl-{ds}", mode=m, row="stage", name=st, p50_ms=v["p50_ms"], p95_ms=v["p95_ms"], mean_ms=v["mean_ms"], throughput_qps="",
                             overhead_vs_base="", gpu_peak_GB=J["memory"][st]["gpu_peak_GB"], host_hwm_delta_GB=J["memory"][st]["host_hwm_delta_GB"]))
    res["datasets"][f"tkgl-{ds}"] = out
res["stage_map"] = {"S1 base scoring": "S1_snapshot (history evolution + snapshot graph build, once per timestamp) + S1_decode (full log-prob row) + S1_top (top-200)",
                    "S2 candidate generation": "S2_HR (top-50 of base row) / S2_PAT (top-50 ++ <=50 pattern-prior block, python per query)",
                    "S3 feature extraction": "S3_HR (19 features) / S3_PAT (25 features)", "S4 re-ranker + fusion": "S4_HR / S4_PAT (seed-0 MLP, calibrated alpha/beta fusion, list order)"}
res["process_peaks"] = {"nvidia_smi_gpu1_peak_MiB": {"tkgl-polecat": 4762, "tkgl-smallpedia": 5104},
                        "host_rss_peak_GB": {k: res["datasets"][k]["memory"]["host_rss_end_GB"]["VmHWM"] for k in res["datasets"]}}
res["concurrent_load"] = {"gpu0": "G8 CognTKE training: 73,472 MiB, mean util 58% (polecat run) / 18% (smallpedia run), sampled every 0.5 s",
                          "host_load_avg_1min": {"tkgl-polecat": 4.79, "tkgl-smallpedia": 5.61}, "host_threads": 48, "host_ram_GB": 503, "docker_cpus": 8}
res["hardware_software"] = dict(host="serverA", gpu="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (GPU 1 only, 96 GB)", driver="580.178.04",
                                cpu="AMD Ryzen Threadripper PRO 7965WX 24-Cores (48 threads)", container="tkg-gpu:latest", python="3.12.3", torch="2.8.0+cu128",
                                cuda="12.8", numpy="2.5.0", scipy="1.18.0")
res["not_run"] = {"tkgl-icews RE-GCN": "not run: the Stage B pipeline (pi_*) uses a different base implementation and feature code; an online adapter would exceed the registered <1 h budget"}
res["caveats"] = ["shared host: G8 training on GPU 0 and other jobs ran concurrently, so absolute latencies are upper bounds for a dedicated machine",
                  "research code (python loops), not a serving implementation; the HRCOV_PAT pattern block (S2_PAT) is a per-query python loop and dominates batched HRCOV_PAT time",
                  "batched mode groups only the sampled queries of a timestamp (polecat mean 33 per batch, smallpedia 588); full-traffic batches would be larger (polecat ~1,752 per timestamp), so batched GPU-stage costs per query are conservative",
                  "end_to_end includes the per-timestamp history evolution amortised over the full test traffic (ratio of sums); excluding it gives throughput_qps_excl_snapshot",
                  "agreement with the stored test lists is <100% because the base rows are re-computed online on a different GPU/batch composition (near-tie order changes); top-1 and gold position agree on >=99.99%"]
json.dump(res, open(f"{OUT}/g6_latency.json", "w"), indent=1)
with open(f"{OUT}/g6_latency_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
for ds, o in res["datasets"].items():
    for m in ("b1", "batched"):
        e = o["end_to_end"][m]
        print(ds, m, " | ".join(f"{a}: p50 {v['p50_ms']:.3f} p95 {v['p95_ms']:.3f} mean {v['mean_ms']:.3f} ms, {v['throughput_qps']:.0f} q/s, ovh {v.get('overhead_vs_base', 0):+.2f}" for a, v in e.items()),
              f"| am {o['snapshot'][m]['amortised_ms_per_query']:.4f} | patshare {e['HRCOV_PAT']['pattern_block_share_of_e2e']:.2f} | ovh_excl HR {e['HR']['overhead_vs_base_excl_snapshot']:+.2f} PAT {e['HRCOV_PAT']['overhead_vs_base_excl_snapshot']:+.2f}")
