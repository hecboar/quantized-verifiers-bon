"""Command-line entry point: python -m vq.cli <command> ...

CPU commands : prepare-data, plan, annotate, gate, analyze
GPU commands : pilot, generate, score, select-lambda, measure-memory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "configs" / "experiment.yaml"
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# Planning assumptions until the pilot writes outputs/throughput.json
DEFAULT_THROUGHPUT = {
    "score": {"sky-1.7b": 20_000, "sky-4b": 10_000, "sky-8b": 5_000, "prm-7b": 5_000},
    "generate": 1_500,
    "avg_score_tokens": 850,
    "avg_gen_tokens": 650,
    "quant_hours": {"gptq": 0.3, "awq": 0.2, "nf4": 0.02, "rtn": 0.02, "none": 0.0},
}


# ------------------------------------------------------------ prepare-data
def default_loaders():
    from . import data as d

    return {
        "gsm8k_test": lambda: d.load_gsm8k("test"),
        "gsm8k_train": lambda: d.load_gsm8k("train"),
        "math500": d.load_math500,
        "math_train": d.load_math_train,
    }


def prepare_data(cfg: dict, loaders: dict, outdir: Path) -> None:
    from .data import make_manifest, save_items, save_manifest

    outdir = Path(outdir)
    if (outdir / "manifest_eval.json").exists():
        raise FileExistsError("manifests already exist; they are frozen by design")
    seed = cfg["seeds"]["manifest"]
    src = {k: f() for k, f in loaders.items()}
    ev_cfg, dev_cfg, dcfg = cfg["data"]["eval"], cfg["data"]["dev"], cfg["data"]

    ev = make_manifest(src["gsm8k_test"], ev_cfg["gsm8k_test"], seed, "gsm8k_eval")["ids"]
    ev += make_manifest(src["math500"], ev_cfg["math500"], seed, "math500_eval")["ids"]
    dev = make_manifest(src["gsm8k_train"], dev_cfg["gsm8k_train"], seed, "gsm8k_dev")["ids"]
    dev += make_manifest(src["math_train"], dev_cfg["math_train"], seed, "math_dev")["ids"]
    if set(ev) & set(dev):
        raise RuntimeError("evaluation and development overlap")

    all_items = [it for items in src.values() for it in items]
    by_id = {it.id: it for it in all_items}
    ev_items, dev_items = [by_id[i] for i in ev], [by_id[i] for i in dev]
    manifests = {
        "eval": make_manifest(ev_items, len(ev), seed, "eval"),
        "dev": make_manifest(dev_items, len(dev), seed, "dev"),
        "gen2": make_manifest(ev_items, dcfg["gen2_subset"], seed + 1, "gen2"),
        "gate": make_manifest(dev_items, dcfg["gate_questions"], seed + 2, "gate"),
    }
    for k, m in manifests.items():
        save_manifest(m, outdir / f"manifest_{k}.json")
    save_items(ev_items + dev_items, outdir / "items.jsonl")


def extend_math_eval(cfg: dict, loaders: dict, outdir: Path) -> None:
    """Preregistered on 2026-09-28 (docs/preregistration.md), before any
    evaluation candidate was generated: +N MATH test problems outside MATH500,
    where the phase-1 dev signal located the quantization damage. The frozen
    manifest_eval.json is left untouched; manifest_eval_full.json = eval + ext."""
    from .data import _hash_ids, load_items, make_manifest, norm_text, save_items, save_manifest

    outdir = Path(outdir)
    if (outdir / "manifest_eval_mathext.json").exists():
        raise FileExistsError("MATH extension already frozen")
    n = cfg["data"]["eval_extension"]["math_test"]
    seed = cfg["seeds"]["manifest"] + 3
    math500_texts = {norm_text(it.question) for it in loaders["math500"]()}
    pool = [it for it in loaders["math_test"]() if norm_text(it.question) not in math500_texts]
    ext = make_manifest(pool, n, seed, "eval_mathext")
    save_manifest(ext, outdir / "manifest_eval_mathext.json")
    base = json.loads((outdir / "manifest_eval.json").read_text())["ids"]
    full_ids = sorted(set(base) | set(ext["ids"]))
    save_manifest({"name": "eval_full", "seed": None, "n": len(full_ids), "ids": full_ids,
                   "sha256": _hash_ids(full_ids), "parts": ["eval", "eval_mathext"]},
                  outdir / "manifest_eval_full.json")
    by_id = {it.id: it for it in pool}
    items = load_items(outdir / "items.jsonl")
    known = {it.id for it in items}
    save_items(items + [by_id[i] for i in ext["ids"] if i not in known], outdir / "items.jsonl")


def eval_manifest_for(cfg: dict, split: str) -> str:
    """The evaluation pools are generated from the preregistered full manifest."""
    return cfg["data"].get("eval_manifest", "eval") if split == "eval" else split


# ------------------------------------------------------------------- plan
def plan_campaigns(cfg: dict, generator: str, tp: dict):
    from .budget import Campaign
    from .experiment import phase_runs

    d = cfg["data"]
    n_eval = sum(d["eval"].values()) + sum(d.get("eval_extension", {}).values())
    C = cfg["generation"]["candidates_per_question"]
    seqs_run = (n_eval + d["dev_scored_per_condition"]) * C
    seqs_gen2 = (d["gen2_subset"] + d["dev_scored_per_condition"]) * C
    out = {}
    for ph in ("phase3_main", "phase4_gen2", "phase5_ch"):
        runs = phase_runs(cfg, ph, generator)
        camps = []
        for r in runs:
            seqs = seqs_gen2 if ph == "phase4_gen2" else seqs_run
            tokens = seqs * tp["avg_score_tokens"]
            # express quantization time as equivalent scoring tokens
            tokens += int(tp["quant_hours"][r.method] * 3600 * tp["score"][r.verifier])
            camps.append(Campaign(r.name, tokens, f"score:{r.verifier}"))
        out[ph] = (runs, camps)
    gen_tokens = (n_eval + sum(d["dev"].values())) * C * tp["avg_gen_tokens"]
    gen_tokens += d["gen2_subset"] * C * tp["avg_gen_tokens"]
    out["phase2_generation"] = ([], [Campaign("generation", gen_tokens, "generate")])

    # prerequisites and validation steps that the grid alone does not show
    tok = tp["avg_score_tokens"]
    dev_full = sum(d["dev"].values()) * C
    devsub = d["dev_scored_per_condition"] * C
    quant = lambda v, m="gptq": int(tp["quant_hours"][m] * 3600 * tp["score"][v])
    hours = lambda name, h: Campaign(name, int(h * 3600), "hours")
    verifiers = cfg["verifiers"]
    out["phase0_pilot"] = ([], [hours("pilot", 1.0)])
    out["prereq_dev_full_bf16"] = ([], [Campaign(f"dev-full-bf16:{v}", dev_full * tok, f"score:{v}") for v in verifiers])
    out["phase1_gate"] = ([], [Campaign(f"gate:{c}", dev_full * tok + (quant("sky-4b") if c == "gptq-w4" else 0),
                                        "score:sky-4b") for c in ("nf4", "gptq-w4")])
    out["lambda_selection"] = ([], [Campaign("devsub-bf16", devsub * tok, "score:sky-4b")] + [
        Campaign(f"lamsel-{lam}", devsub * tok + quant("sky-4b"), "score:sky-4b")
        for lam in cfg["calibration"]["lambda_grid"]])
    orms = [v for v in verifiers if v.startswith("sky-")]
    dep = [Campaign(f"nf4-bnb:{v}", seqs_run * tok, f"score:{v}") for v in orms]
    dep += [Campaign(f"gptq-w4-packed:{v}", seqs_run * tok, f"score:{v}") for v in ("sky-4b", "sky-8b")]
    dep += [Campaign(f"export-packed:{v}", quant(v), f"score:{v}") for v in ("sky-4b", "sky-8b")]
    dep += [hours("measure-memory", 0.5)]
    out["phase6_deployment"] = ([], dep)
    # each pod launch: dependency install (~15 min) + upload/fetch (~6 min)
    n_launches = 12
    out["pod_overhead"] = ([], [hours(f"launch-{i}", 0.35) for i in range(n_launches)])
    return out


def cmd_plan(args) -> None:
    from .budget import forecast
    from .experiment import load_config

    cfg = load_config(args.config)
    tp = DEFAULT_THROUGHPUT
    tp_file = OUT / "throughput.json"
    if tp_file.exists():
        tp = {**tp, **json.loads(tp_file.read_text())}
        print(f"using measured throughput from {tp_file}")
    else:
        print("using PLANNING ASSUMPTIONS (run `pilot` on GPU to measure)")
    rates = {"generate": tp["generate"], "hours": 1.0, **{f"score:{v}": r for v, r in tp["score"].items()}}
    b = cfg["budget"]
    total = 0.0
    for ph, (runs, camps) in plan_campaigns(cfg, args.generator, tp).items():
        f = forecast(camps, rates, b["eur_per_hour"], 0.0 if ph in ("pod_overhead", "phase0_pilot") else b["overhead_frac"])
        total += f["eur"]
        n = f"{len(runs)} runs" if runs else f"{len(camps)} steps"
        print(f"{ph}: {n}, {f['hours']:.1f} h, {f['eur']:.1f} EUR")
    flag = "OVER CEILING" if total > b["submission_ceiling_eur"] else "within ceiling"
    print(f"TOTAL (all paid steps): {total:.1f} EUR vs submission ceiling {b['submission_ceiling_eur']} EUR -> {flag}")


# ---------------------------------------------------------------- helpers
def _items():
    from .data import load_items

    return {it.id: it for it in load_items(DATA / "items.jsonl")}


def _manifest(name):
    from .data import load_manifest

    return load_manifest(DATA / f"manifest_{name}.json")["ids"]


def _store():
    from .store import CampaignStore

    return CampaignStore(OUT / "store")


def _load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


# ------------------------------------------------------------- generation
def cmd_generate(args) -> None:
    from .experiment import load_config
    from .generate import GENERATORS, generate_pool

    cfg, items, store = load_config(args.config), _items(), _store()
    g = cfg["generation"]
    manifest = eval_manifest_for(cfg, args.split)
    ids = _manifest(manifest)[: args.limit] if args.limit else _manifest(manifest)
    # the secondary generator's gen2 subset plays the role of its evaluation pools
    split_name = "eval" if args.split == "gen2" else args.split
    name = args.name or (f"{args.generator}__{split_name}" + (f"-pilot{args.limit}" if args.limit else ""))
    if store.done("raw_candidates", name):
        if args.fresh:
            sys.exit(f"{name} already exists but --fresh was requested")
        print(f"{name} exists, skipping")
        return
    generate_pool(GENERATORS[args.generator], [items[i] for i in ids], store, name, tuple(g["seeds"]),
                  g["per_seed"], g["temperature"], g["top_p"], g["max_tokens"],
                  meta={"generator": args.generator, "split": args.split, **g})


def cmd_annotate(args) -> None:
    from .generate import annotate

    items, store = _items(), _store()
    for name in store.list("raw_candidates"):
        if args.generator and not name.startswith(args.generator + "__"):
            continue
        if store.done("candidates", name):
            continue
        df = annotate(store.read("raw_candidates", name), items)
        store.write("candidates", name, df, meta=store.meta("raw_candidates", name))
        print(f"annotated {name}: acc={df['correct'].mean():.3f} parsed={df['parsed'].mean():.3f} "
              f"truncated={df['truncated'].mean():.3f}")


# ---------------------------------------------------------------- scoring
def _context(args):
    from .runner import RunContext
    from .verifiers import VERIFIERS

    items = _items()
    lam_file = OUT / "frozen_lambda.json"
    return RunContext(
        store=_store(), questions={k: v.question for k, v in items.items()}, device="cuda",
        verifier_specs=VERIFIERS, generic_texts=_generic_texts(), n_calib=128,
        dev_scored_ids=_manifest("gate"),  # frozen stratified dev subset scored in every condition
        frozen_lambda=json.loads(lam_file.read_text()) if lam_file.exists() else {},
    )


def _generic_texts() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")  # namespaced id (newer hub requires it)
    return [t for t in ds["text"] if len(t) > 200]


def cmd_score(args) -> None:
    from .experiment import ScoreRun, load_config, phase_runs
    from .runner import execute_run

    cfg, ctx = load_config(args.config), _context(args)
    runs = phase_runs(cfg, args.phase, args.generator)
    if args.only:
        runs = [r for r in runs if r.verifier in args.only.split(",")]
    if args.conditions:
        runs = [r for r in runs if r.condition in args.conditions.split(",")]
    # BF16 dev pools of the calibration generator first: recipes depend on them
    for gen, v in sorted({(r.calib_gen, r.verifier) for r in runs}):
        execute_run(ScoreRun(gen, v, "bf16", "none"), ctx, dev_full=True)
    for r in runs:
        # --dev-only: phase-1 gate, before evaluation pools exist
        status = execute_run(r, ctx, dev_full=args.dev_only)
        print(f"[{args.phase}{' dev-only' if args.dev_only else ''}] {r.name}: {status}", flush=True)


def cmd_phase1(args) -> None:
    """Phase 1 in one pod step: generate the dev pools (child process: vLLM frees
    its VRAM on exit), annotate, score the phase1_signal grid on dev, and write
    the early signal (quantization vs BF16 next to the BF16 null floor)."""
    import subprocess

    from .analysis import early_signal
    from .experiment import ScoreRun, load_config, phase_runs
    from .runner import execute_run

    cfg = load_config(args.config)
    name = f"{args.generator}__dev"
    gen = [sys.executable, "-m", "vq.cli", "generate", "--generator", args.generator, "--split", "dev",
           "--name", name]
    if args.limit:
        gen += ["--limit", str(args.limit)]
    subprocess.run(gen, check=True)
    cmd_annotate(argparse.Namespace(generator=args.generator, config=args.config))
    ctx = _context(args)
    runs = phase_runs(cfg, "phase1_signal", args.generator)
    for v in sorted({r.verifier for r in runs}):
        execute_run(ScoreRun(args.generator, v, "bf16", "none"), ctx, dev_full=True)
    for r in runs:
        print(f"[phase1 dev] {r.verifier} {r.condition}: {execute_run(r, ctx, dev_full=True)}", flush=True)
    verifiers = sorted({r.verifier for r in runs})
    conds = [f"dev-full-{c}" for c in cfg["phases"]["phase1_signal"]["grid"]["conditions"] if c != "bf16"]
    sig = early_signal(ctx.store, args.generator, verifiers, conds, split="dev", reference="dev-full-bf16")
    sig["_generation"] = ctx.store.meta("raw_candidates", name).get("throughput")
    OUT.mkdir(exist_ok=True)
    (OUT / "signal.json").write_text(json.dumps(sig, indent=1))
    print("EARLY SIGNAL (dev, N = full pool): delta accuracy [95% CI], winner change, noise floor")
    for v in verifiers:
        floor = sig[v]["noise_floor_winner_change"]
        for c in conds:
            r = sig[v][c]
            print(f"  {v:<8} {c.replace('dev-full-', ''):<10} dA={r['delta_acc']:+.3f} "
                  f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]  winner change={r['winner_change']:.3f}  "
                  f"(floor {floor:.3f})")


def cmd_select_lambda(args) -> None:
    """CH lambda chosen on the frozen dev-scored subset only (no eval scoring),
    then frozen for every CH run (plan §5)."""
    from .analysis import score_matrix
    from .experiment import ScoreRun, load_config
    from .runner import execute_run, select_lambda

    cfg, ctx = load_config(args.config), _context(args)
    execute_run(ScoreRun(args.generator, args.verifier, "bf16", "none"), ctx, dev_full=True)  # CH pair source
    execute_run(ScoreRun(args.generator, args.verifier, "bf16", "none"), ctx, dev_scored_only=True)
    ref, *_ = score_matrix(ctx.store, args.generator, args.verifier, "devsub-bf16", "dev")
    cands = {}
    for lam in cfg["calibration"]["lambda_grid"]:
        run = ScoreRun(args.generator, args.verifier, f"lamsel-{lam}", "gptq", 4, 128, "ch", 0, lam)
        execute_run(run, ctx, dev_scored_only=True)
        cands[lam], *_ = score_matrix(ctx.store, args.generator, args.verifier, f"devsub-lamsel-{lam}", "dev")
    best = select_lambda(ref, cands)
    lam_file = OUT / "frozen_lambda.json"
    frozen = json.loads(lam_file.read_text()) if lam_file.exists() else {}
    if frozen and any(frozen.get(v) not in (None, best) for v in ("sky-4b", "sky-8b")):
        sys.exit(f"refusing to overwrite a different frozen lambda {frozen} with {best}")
    # CH-GPTQ for both ORMs reuses the lambda chosen on the reference model
    frozen.update({args.verifier: best, "sky-4b": best, "sky-8b": best})
    lam_file.write_text(json.dumps(frozen, indent=1))
    print(f"frozen lambda: {frozen}")


# ------------------------------------------------------------------ pilot
def cmd_pilot(args) -> None:
    """Phase 0: throughput + integrity checks on a handful of dev questions."""
    from . import verifiers as V
    from .pilot import run_pilot
    from .runner import generic_chunks

    items = _items()
    ids = _manifest("gate")[: args.questions]
    texts = _generic_texts()
    specs = {k: V.VERIFIERS[k] for k in ("sky-1.7b", "sky-4b", "sky-8b", "prm-7b")}
    def generate_in_subprocess(its):
        # vLLM keeps ~90% of VRAM reserved until its process exits: generate in a
        # child process so the verifiers can be loaded afterwards
        import subprocess

        import time as _t

        name = f"{args.generator}__gate-pilot{args.questions}-{int(_t.time())}"  # fresh every pilot
        subprocess.run([sys.executable, "-m", "vq.cli", "generate", "--generator", args.generator,
                        "--split", "gate", "--limit", str(args.questions), "--name", name, "--fresh"], check=True)
        st = _store()
        tps = st.meta("raw_candidates", name).get("throughput", {}).get("tok_per_s")
        df = st.read("raw_candidates", name)
        return (df, tps) if tps else df

    rep = run_pilot(
        items, ids, generate_in_subprocess, V.load_verifier, specs,
        quant_verifier="sky-1.7b", n_calib=16, device=args.device,
        generic_chunk=lambda tok, n: generic_chunks(tok, texts, args.chunk_tokens, n),
    )
    rep["generator"] = args.generator
    qc = rep["quant_check"].get("sky-1.7b")
    if qc:  # GPTQ time for the grid: 128 calibration sequences, scaled from 1.7B to ~8B parameters
        per_seq = qc["per_calib_seq_seconds"]
        rep["quant_hours"] = {"gptq": per_seq * 128 * 4.7 / 3600, "awq": per_seq * 128 * 3 / 3600,
                              "nf4": 0.02, "rtn": 0.02, "none": 0.0}
    OUT.mkdir(exist_ok=True)
    # the report is always saved and printed, errors included (never lose the diagnosis)
    (OUT / f"pilot_{args.generator}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["ok"]:
        (OUT / "throughput.json").write_text(json.dumps(rep, indent=1))  # only a clean pilot feeds `plan`
        print("PILOT OK")
    else:
        print("PILOT FAILED: do not launch further phases")
        for v, err in rep["errors"].items():
            print(f"  {v}: {err}")
        sys.exit(1)


# --------------------------------------------------------- memory (C2)
def cmd_export_packed(args) -> None:
    """GPTQ-W4 (generic calibration, same recipe as the grid) -> packed checkpoint."""
    import torch

    from .calibration import build_recipe
    from .quant.apply import quantize_model
    from .quant.export import export_gptq_checkpoint
    from .runner import generic_chunks
    from .verifiers import VERIFIERS, load_verifier

    spec = VERIFIERS[args.verifier]
    tok, model = load_verifier(spec)
    ids = generic_chunks(tok, _generic_texts(), 512, 256)
    calib = build_recipe("generic", None, 128, 0, generic_texts_ids=ids)
    stats = quantize_model(model, "gptq", 4, 128, calib=calib, keep_params=True)
    pdir = OUT / "packed" / f"{args.verifier}__gptq-w4"
    export_gptq_checkpoint(model, stats["params"], pdir, bits=4, group_size=128)
    tok.save_pretrained(pdir)
    del model
    torch.cuda.empty_cache()
    print(f"exported {pdir}")


def cmd_score_deployed(args) -> None:
    """Full scoring with the real deployment (bitsandbytes NF4 or packed GPTQ
    kernels): the memory frontier uses these scores, never fake-quant ones."""
    import torch

    from .experiment import ScoreRun
    from .runner import execute_run
    from .verifiers import load_verifier_deployed

    ctx = _context(args)
    pdir = str(OUT / "packed" / f"{args.verifier}__gptq-w4")
    cond = {"nf4": "nf4-bnb", "gptq-w4": "gptq-w4-packed"}[args.scheme]
    ctx.dtype = torch.bfloat16 if args.scheme == "nf4" else torch.float16
    ctx.loader = lambda sp, device="cuda", dtype=None: load_verifier_deployed(sp, args.scheme, device, dtype, pdir)
    print(execute_run(ScoreRun(args.generator, args.verifier, cond, "none"), ctx))


def cmd_parity(args) -> None:
    from .analysis import compare_conditions

    rep = compare_conditions(_store(), args.generator, args.verifier, args.a, args.b, args.split)
    (OUT / f"parity_{args.verifier}_{args.a}_vs_{args.b}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def cmd_measure_memory(args) -> None:
    from .memory_bench import measure_all

    report = measure_all(args.verifiers.split(","), OUT / "store", OUT / "packed")
    (OUT / "memory.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ----------------------------------------------------------- CPU analysis
def cmd_gate(args) -> None:
    from .analysis import condition_metrics, evaluate_gate, mixed_fraction, score_matrix

    store = _store()
    # phase 1 runs before evaluation pools exist: dev-only BF16 scores
    gate_q = _manifest("gate")  # the pre-registered 100 gate questions, not all dev
    _, correct, _, _ = score_matrix(store, args.generator, args.verifier, "dev-full-bf16", "dev", questions=gate_q)
    n = correct.shape[1]  # full pool (32 in the experiment)
    m = condition_metrics(store, args.generator, args.verifier, "dev-full-bf16", budgets=[n], split="dev",
                          vote_temperature=None, reference="dev-full-bf16", questions=gate_q)
    res = evaluate_gate(
        acc_bf16_32=float(m[n]["acc_ref"].mean()), acc_random_32=float(m[n]["random"].mean()),
        mixed_frac=mixed_fraction(correct), projected_eur=args.projected_eur, implementation_ok=args.implementation_ok,
    )
    res.update(generator=args.generator, verifier=args.verifier, questions=len(gate_q))
    OUT.mkdir(exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(res, indent=1))  # scripts/pod.py requires go=true for grid steps
    print(json.dumps(res, indent=1))
    if not res["go"]:
        sys.exit(1)


def cmd_analyze(args) -> None:
    from .report import write_report

    write_report(_store(), args.generator, OUT / "tables")


# ------------------------------------------------------------------- main
def main(argv=None) -> None:
    _load_env()
    p = argparse.ArgumentParser(prog="vq")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=str(DEFAULT_CFG))
        sp.set_defaults(fn=fn)
        return sp

    add("extend-math", lambda a: extend_math_eval(
        __import__("vq.experiment", fromlist=["x"]).load_config(a.config),
        {"math_test": __import__("vq.data", fromlist=["x"]).load_math_test,
         "math500": __import__("vq.data", fromlist=["x"]).load_math500}, DATA))
    add("prepare-data", lambda a: prepare_data(__import__("vq.experiment", fromlist=["x"]).load_config(a.config),
                                               default_loaders(), DATA))
    sp = add("plan", cmd_plan)
    sp.add_argument("--generator", default="qwen2.5-1.5b")
    sp = add("generate", cmd_generate)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--split", choices=["eval", "dev", "gen2", "gate"], required=True)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--name", default=None, help="store name override (the pilot uses a unique one)")
    sp.add_argument("--fresh", action="store_true", help="fail instead of reusing existing candidates")
    sp = add("annotate", cmd_annotate)
    sp.add_argument("--generator", default=None)
    sp = add("score", cmd_score)
    sp.add_argument("--phase", required=True)
    sp.add_argument("--generator", default=None)
    sp.add_argument("--only", default=None, help="comma-separated verifier keys")
    sp.add_argument("--conditions", default=None, help="comma-separated condition names")
    sp.add_argument("--dev-only", action="store_true", help="score development pools only (phase-1 gate)")
    sp = add("phase1", cmd_phase1)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--limit", type=int, default=None, help="dev questions (tests only; default all 400)")
    sp = add("select-lambda", cmd_select_lambda)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--verifier", default="sky-4b")
    sp = add("pilot", cmd_pilot)
    sp.add_argument("--generator", default="qwen2.5-1.5b")
    sp.add_argument("--questions", type=int, default=16)
    sp.add_argument("--device", default="cuda", help="cpu only for the local end-to-end test")
    sp.add_argument("--chunk-tokens", type=int, default=512)
    sp = add("export-packed", cmd_export_packed)
    sp.add_argument("--verifier", required=True)
    sp = add("score-deployed", cmd_score_deployed)
    sp.add_argument("--verifier", required=True)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--scheme", choices=["nf4", "gptq-w4"], required=True)
    sp = add("parity", cmd_parity)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--verifier", required=True)
    sp.add_argument("--a", required=True, help="e.g. nf4 (fake-quant)")
    sp.add_argument("--b", required=True, help="e.g. nf4-bnb (deployment)")
    sp.add_argument("--split", default="dev")
    sp = add("measure-memory", cmd_measure_memory)
    sp.add_argument("--verifiers", default="sky-1.7b,sky-4b,sky-8b,prm-7b")
    sp = add("gate", cmd_gate)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--verifier", default="sky-4b")
    sp.add_argument("--projected-eur", type=float, required=True)
    sp.add_argument("--implementation-ok", action="store_true")
    sp = add("analyze", cmd_analyze)
    sp.add_argument("--generator", required=True)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
