"""Find which ONNX node DirectML computes wrong (AnyStemDeck diagnostic).

Context: `scripts/test_dml_smoke.py --split` produces plausible-looking but
too-loud output on DirectML (peaks in the tens, vs ~0.1-0.7 for the
identical code running on CPU -- see docs/plan.md). Since the same graph,
same code, same input produces correct results on CPU (`--provider cpu`)
and wrong results on DirectML (`--provider dml`), the divergence is
somewhere inside ONNX Runtime's DirectML execution provider itself --
some node computes a numerically different (not just imprecise-but-close)
result on DML than on CPU.

v2 -- binary search instead of "instrument everything at once". The first
version added all ~3500 intermediate tensors as graph outputs in one shot,
which choked DirectML's own graph partitioning (it couldn't bind a few of
those synthetic outputs -- especially pure `Constant` node outputs -- to
the DML EP and silently fell back to CPU-only, invalidating the whole
comparison). This version adds exactly ONE extra output per probe and
binary-searches the graph's node list (in execution order) for the first
point where CPU and DirectML disagree -- ~12 probes for a ~3500-node
graph instead of one fragile 3500-output run, and each probe is small
enough that a bad node just gets skipped rather than derailing everything.

Usage (needs the actual DirectML machine):

    python scripts/diff_dml_nodes.py
    python scripts/diff_dml_nodes.py --model htdemucs_6s --threshold 0.05
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Node types whose output can't meaningfully "diverge" per-input (pure
# compile-time constants / shape bookkeeping) -- also the ones that
# tripped DirectML's graph partitioning when forced into the output list
# in v1. Skipped as binary-search candidates entirely.
_SKIP_OP_TYPES = {"Constant", "ConstantOfShape", "Shape"}


def _candidate_outputs(model: onnx.ModelProto) -> list[tuple[str, str]]:
    """Every non-skipped node's (output_name, op_type), in execution order.
    A node with multiple outputs contributes its first non-empty one."""
    out = []
    for node in model.graph.node:
        if node.op_type in _SKIP_OP_TYPES:
            continue
        for name in node.output:
            if name:
                out.append((name, node.op_type))
                break
    return out


def _run_with_extra_output(base_path: Path, extra_output: str, provider: str,
                            dummy: np.ndarray, workdir: Path) -> np.ndarray | None:
    """Load `base_path`, add `extra_output` as an additional graph output if
    it isn't one already, run with `dummy`, return that one output's array
    (or None if this EP can't produce it -- treated as "skip this probe",
    not a hard failure, since a handful of node types are known to be
    awkward to force as standalone outputs on some EPs)."""
    model = onnx.load(str(base_path))
    existing = {o.name for o in model.graph.output}
    if extra_output not in existing:
        inferred = onnx.shape_inference.infer_shapes(model)
        vi_by_name = {vi.name: vi for vi in inferred.graph.value_info}
        vi = vi_by_name.get(extra_output)
        if vi is not None:
            model.graph.output.append(vi)
        else:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(extra_output, onnx.TensorProto.FLOAT, None),
            )

    probe_path = workdir / f"probe_{provider}.onnx"
    onnx.save(model, str(probe_path))

    ort_provider = "DmlExecutionProvider" if provider == "dml" else "CPUExecutionProvider"
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if provider == "dml":
        sess_opts.enable_mem_pattern = False
    try:
        sess = ort.InferenceSession(str(probe_path), sess_options=sess_opts, providers=[ort_provider])
        outputs = [o.name for o in sess.get_outputs()]
        results = sess.run(outputs, {"mix": dummy})
        return dict(zip(outputs, results))[extra_output]
    except Exception as e:  # noqa: BLE001 -- diagnostic tool, report and move on
        print(f"    ({provider} couldn't produce {extra_output!r}: {e.__class__.__name__}; skipping this probe)")
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="htdemucs_6s")
    p.add_argument("--threshold", type=float, default=0.05,
                    help="Nodes diverging by more than this (max abs diff) count as 'wrong' (default 0.05)")
    args = p.parse_args()

    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("DmlExecutionProvider not available -- this diagnostic needs to run on the DirectML machine.")
        sys.exit(1)

    from app.pipeline.onnx_export import export_split_to_onnx
    from app.pipeline.onnx_export.export_split import N_SAMPLES

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_path = tmp / "split.onnx"
        print(f"Exporting split graph for {args.model!r} (once -- reused for every probe)...")
        export_split_to_onnx(args.model, split_path, verbose=False)

        model = onnx.load(str(split_path))
        candidates = _candidate_outputs(model)
        print(f"{len(candidates)} candidate nodes (execution order) to binary-search over.")

        torch.manual_seed(0)
        dummy = torch.randn(1, 2, N_SAMPLES, dtype=torch.float32).numpy()

        def diff_at(idx: int) -> float | None:
            name, op_type = candidates[idx]
            print(f"  probing node #{idx}/{len(candidates)}  op_type={op_type}  output={name!r}")
            cpu_val = _run_with_extra_output(split_path, name, "cpu", dummy, tmp)
            dml_val = _run_with_extra_output(split_path, name, "dml", dummy, tmp)
            if cpu_val is None or dml_val is None:
                return None
            if cpu_val.shape != dml_val.shape or cpu_val.size == 0:
                print(f"    shape mismatch or empty ({cpu_val.shape} vs {dml_val.shape}); skipping")
                return None
            diff = float(np.abs(cpu_val.astype(np.float64) - dml_val.astype(np.float64)).max())
            print(f"    max_abs_diff={diff:.6g}")
            return diff

        # Binary search execution order for the first index where CPU and
        # DML disagree beyond `threshold`. Not a strictly monotonic
        # property in a general DAG (independent branches can interleave
        # in node order), but htdemucs's forward pass is close enough to
        # linear (freq branch / time branch / crosstransformer / decoder,
        # each mostly sequential) that this converges on a genuinely
        # useful answer in practice -- verify the reported node's
        # immediate neighbors too before concluding.
        lo, hi = 0, len(candidates) - 1
        first_bad = None
        probes = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            d = diff_at(mid)
            probes += 1
            if d is None:
                # Unusable probe -- nudge inward and keep going rather than
                # get stuck (rare; only a handful of op types hit this).
                hi = mid - 1
                continue
            if d > args.threshold:
                first_bad = mid
                hi = mid - 1
            else:
                lo = mid + 1

        print(f"\n{'=' * 70}\nRESULT ({probes} probes)\n{'=' * 70}")
        if first_bad is None:
            print("No divergent node found via binary search. Either DirectML matches "
                  "CPU throughout this graph for this input (unlikely, given the smoke "
                  "test's real-song result), or the divergence is in one of the skipped "
                  f"op types ({sorted(_SKIP_OP_TYPES)}), or it only shows up on real audio, "
                  "not this random dummy -- worth adapting this script to load a real "
                  "audio chunk instead if so.")
        else:
            name, op_type = candidates[first_bad]
            print(f"First divergent node (execution order): #{first_bad}  op_type={op_type}  output={name!r}")
            print("\nSanity-checking neighbors...")
            for j in (max(0, first_bad - 1), min(len(candidates) - 1, first_bad + 1)):
                if j != first_bad:
                    diff_at(j)


if __name__ == "__main__":
    main()
