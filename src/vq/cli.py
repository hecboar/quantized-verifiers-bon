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


# ------------------------------------------------------------------- plan
def plan_campaigns(cfg: dict, generator: str, tp: dict):
    from .budget import Campaign
    from .experiment import phase_runs

    d = cfg["data"]
    n_eval = sum(d["eval"].values())
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
    rates = {"generate": tp["generate"], **{f"score:{v}": r for v, r in tp["score"].items()}}
    b = cfg["budget"]
    total = 0.0
    for ph, (runs, camps) in plan_campaigns(cfg, args.generator, tp).items():
        f = forecast(camps, rates, b["eur_per_hour"], b["overhead_frac"])
        total += f["eur"]
        n = f"{len(runs)} runs" if runs else "generation"
        print(f"{ph}: {n}, {f['hours']:.1f} h, {f['eur']:.1f} EUR")
    flag = "OVER CEILING" if total > b["submission_ceiling_eur"] else "within ceiling"
    print(f"TOTAL (phases 2-5): {total:.1f} EUR vs submission ceiling {b['submission_ceiling_eur']} EUR -> {flag}")


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
    ids = _manifest(args.split)[: args.limit] if args.limit else _manifest(args.split)
    name = f"{args.generator}__{args.split}" + (f"-pilot{args.limit}" if args.limit else "")
    if store.done("raw_candidates", name):
        print(f"{name} exists, skipping")
        return
    df = generate_pool(GENERATORS[args.generator], [items[i] for i in ids], tuple(g["seeds"]), g["per_seed"],
                       g["temperature"], g["top_p"], g["max_tokens"])
    store.write("raw_candidates", name, df, meta={"generator": args.generator, "split": args.split, **g})


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
        frozen_lambda=json.loads(lam_file.read_text()) if lam_file.exists() else {},
    )


def _generic_texts() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return [t for t in ds["text"] if len(t) > 200]


def cmd_score(args) -> None:
    from .experiment import ScoreRun, load_config, phase_runs
    from .runner import execute_run

    cfg, ctx = load_config(args.config), _context(args)
    runs = phase_runs(cfg, args.phase, args.generator)
    if args.only:
        runs = [r for r in runs if r.verifier in args.only.split(",")]
    # BF16 dev pools first: calibration recipes depend on them
    for v in sorted({r.verifier for r in runs}):
        gen = runs[0].generator
        execute_run(ScoreRun(gen, v, "dev-full-bf16", "none"), ctx, dev_full=True)
    for r in runs:
        print(f"[{args.phase}] {r.name}: {execute_run(r, ctx)}", flush=True)


def cmd_select_lambda(args) -> None:
    from .analysis import score_matrix
    from .experiment import ScoreRun, load_config
    from .runner import execute_run, select_lambda

    cfg, ctx = load_config(args.config), _context(args)
    ctx.dev_scored = cfg["data"]["dev_scored_per_condition"]
    execute_run(ScoreRun(args.generator, args.verifier, "dev-full-bf16", "none"), ctx, dev_full=True)
    execute_run(ScoreRun(args.generator, args.verifier, "bf16", "none"), ctx)
    ref, *_ = score_matrix(ctx.store, args.generator, args.verifier, "bf16", "dev")
    cands = {}
    for lam in cfg["calibration"]["lambda_grid"]:
        run = ScoreRun(args.generator, args.verifier, f"lamsel-{lam}", "gptq", 4, 128, "ch", 0, lam)
        execute_run(run, ctx)
        cands[lam], *_ = score_matrix(ctx.store, args.generator, args.verifier, f"lamsel-{lam}", "dev")
    best = select_lambda(ref, cands)
    lam_file = OUT / "frozen_lambda.json"
    frozen = json.loads(lam_file.read_text()) if lam_file.exists() else {}
    frozen[args.verifier] = best
    # CH-GPTQ for other verifiers reuses the lambda chosen on the reference model
    for v in ("sky-4b", "sky-8b"):
        frozen.setdefault(v, best)
    lam_file.write_text(json.dumps(frozen, indent=1))
    print(f"frozen lambda: {frozen}")


# ------------------------------------------------------------------ pilot
def cmd_pilot(args) -> None:
    """Phase 0: throughput + integrity checks on a handful of dev questions."""
    import time

    import torch

    from .generate import GENERATORS, annotate, generate_pool
    from .verifiers import VERIFIERS, Scorer, load_verifier

    items = _items()
    ids = _manifest("gate")[: args.questions]
    report = {"score": {}, "parity": {}}
    t0 = time.perf_counter()
    df = generate_pool(GENERATORS[args.generator], [items[i] for i in ids])
    report["generate"] = int(df["n_tokens"].sum() / (time.perf_counter() - t0))
    df = annotate(df, items)
    report["avg_gen_tokens"] = float(df["n_tokens"].mean())
    report["truncated_frac"] = float(df["truncated"].mean())
    report["parsed_frac"] = float(df["parsed"].mean())
    qs = [items[q].question for q in df["question_id"]]
    for v in ("sky-1.7b", "sky-4b", "sky-8b", "prm-7b"):
        tok, model = load_verifier(VERIFIERS[v])
        sc = Scorer(VERIFIERS[v], tok, model)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = sc.score(qs, df["text"].tolist())
        torch.cuda.synchronize()
        report["score"][v] = int(sum(res["n_tokens"]) / (time.perf_counter() - t0))
        report["avg_score_tokens"] = float(sum(res["n_tokens"]) / len(res["n_tokens"]))
        sc.max_tokens = 1
        single = sc.score(qs[:16], df["text"].tolist()[:16])["score"]
        report["parity"][v] = float(max(abs(a - b) for a, b in zip(res["score"][:16], single)))
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(exist_ok=True)
    (OUT / "throughput.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


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


def cmd_score_packed(args) -> None:
    """Full-test scoring with the packed kernel (replaces fake-quant numbers on the frontier)."""
    import torch

    from .experiment import ScoreRun
    from .runner import execute_run
    from .verifiers import VerifierSpec

    ctx = _context(args)
    pdir = OUT / "packed" / f"{args.verifier}__gptq-w4"
    ctx.verifier_specs = {args.verifier: VerifierSpec(args.verifier, str(pdir), "orm")}
    ctx.dtype = torch.float16
    print(execute_run(ScoreRun(args.generator, args.verifier, "gptq-w4-packed", "none"), ctx))


def cmd_measure_memory(args) -> None:
    from .memory_bench import measure_all

    report = measure_all(args.verifiers.split(","), OUT / "store", OUT / "packed")
    (OUT / "memory.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ----------------------------------------------------------- CPU analysis
def cmd_gate(args) -> None:
    from .analysis import condition_metrics, evaluate_gate, mixed_fraction, score_matrix

    store = _store()
    _, correct, _, _ = score_matrix(store, args.generator, args.verifier, "bf16", "dev")
    m = condition_metrics(store, args.generator, args.verifier, "bf16", budgets=[32], split="dev", vote_temperature=None)
    res = evaluate_gate(
        acc_bf16_32=float(m[32]["acc_ref"].mean()), acc_random_32=float(m[32]["random"].mean()),
        mixed_frac=mixed_fraction(correct), projected_eur=args.projected_eur, implementation_ok=args.implementation_ok,
    )
    print(json.dumps(res, indent=1))


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

    add("prepare-data", lambda a: prepare_data(__import__("vq.experiment", fromlist=["x"]).load_config(a.config),
                                               default_loaders(), DATA))
    sp = add("plan", cmd_plan)
    sp.add_argument("--generator", default="qwen2.5-1.5b")
    sp = add("generate", cmd_generate)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--split", choices=["eval", "dev", "gen2", "gate"], required=True)
    sp.add_argument("--limit", type=int, default=None)
    sp = add("annotate", cmd_annotate)
    sp.add_argument("--generator", default=None)
    sp = add("score", cmd_score)
    sp.add_argument("--phase", required=True)
    sp.add_argument("--generator", default=None)
    sp.add_argument("--only", default=None, help="comma-separated verifier keys")
    sp = add("select-lambda", cmd_select_lambda)
    sp.add_argument("--generator", required=True)
    sp.add_argument("--verifier", default="sky-4b")
    sp = add("pilot", cmd_pilot)
    sp.add_argument("--generator", default="qwen2.5-1.5b")
    sp.add_argument("--questions", type=int, default=16)
    sp = add("export-packed", cmd_export_packed)
    sp.add_argument("--verifier", required=True)
    sp = add("score-packed", cmd_score_packed)
    sp.add_argument("--verifier", required=True)
    sp.add_argument("--generator", required=True)
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
