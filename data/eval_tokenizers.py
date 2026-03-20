import os
import time
import json
from datetime import datetime
from bpe_tokenizer import sample_text, load_tokenizer, encode, ARTIFACTS_PATH, TOKENIZER_PATH

BPE_TOKENIZER_PATH = TOKENIZER_PATH
SP_TOKENIZER_PATH = "data/sp-tokenizer"
RESULTS_PATH = "data/eval_results.md"
NUM_EVAL_DOCS = 1000


def run_eval():
    files = sorted([os.path.join(ARTIFACTS_PATH, f) for f in os.listdir(ARTIFACTS_PATH) if f.endswith(".parquet")])

    print(f"Loading {NUM_EVAL_DOCS} eval docs...")
    docs = []
    for i, text in enumerate(sample_text(files)):
        if i >= NUM_EVAL_DOCS:
            break
        docs.append(text)
    total_chars = sum(len(d) for d in docs)
    print(f"{len(docs)} docs, {total_chars:,} chars\n")

    results = {}

    # Custom BPE
    if os.path.exists(os.path.join(BPE_TOKENIZER_PATH, "merges.json")):
        print("Benchmarking custom BPE...")
        merges, vocab = load_tokenizer()
        total_tokens = 0
        failures = 0
        start = time.time()
        for doc in docs:
            try:
                total_tokens += len(encode(doc, merges, vocab))
            except Exception:
                failures += 1
        elapsed = time.time() - start
        results["Custom BPE"] = {
            "vocab_size": len(vocab),
            "total_tokens": total_tokens,
            "tokens_per_char": total_tokens / total_chars,
            "time": elapsed,
            "docs_per_sec": (len(docs) - failures) / elapsed,
            "failures": failures,
        }

    # SentencePiece
    sp_model = os.path.join(SP_TOKENIZER_PATH, "sp_model.model")
    if os.path.exists(sp_model):
        print("Benchmarking SentencePiece...")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=sp_model)
        total_tokens = 0
        start = time.time()
        for doc in docs:
            total_tokens += len(sp.encode(doc, out_type=int))
        elapsed = time.time() - start
        results["SentencePiece"] = {
            "vocab_size": sp.vocab_size(),
            "total_tokens": total_tokens,
            "tokens_per_char": total_tokens / total_chars,
            "time": elapsed,
            "docs_per_sec": len(docs) / elapsed,
            "failures": 0,
        }

    if not results:
        print("No tokenizers found. Train at least one first.")
        return

    # Build markdown
    lines = [
        f"# Tokenizer Evaluation",
        f"",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Eval docs: {len(docs)}  ",
        f"Total chars: {total_chars:,}",
        f"",
        f"## Results",
        f"",
    ]

    names = list(results.keys())
    header = f"| {'Metric':<20} | " + " | ".join(f"{n:>14}" for n in names) + " |"
    sep = f"| {'-'*20} | " + " | ".join("-" * 14 for _ in names) + " |"
    lines.append(header)
    lines.append(sep)

    rows = [
        ("Vocab size",      "vocab_size",     "d"),
        ("Total tokens",    "total_tokens",   ",d"),
        ("Tokens/char",     "tokens_per_char", ".3f"),
        ("Encode time (s)", "time",           ".3f"),
        ("Docs/sec",        "docs_per_sec",   ".1f"),
        ("Failures",        "failures",       "d"),
    ]
    for label, key, fmt in rows:
        row = f"| {label:<20} | "
        row += " | ".join(f"{results[n][key]:>14{fmt}}" for n in names)
        row += " |"
        lines.append(row)

    if len(results) == 2:
        bpe = results["Custom BPE"]
        sp = results["SentencePiece"]
        speedup = sp["docs_per_sec"] / bpe["docs_per_sec"] if bpe["docs_per_sec"] > 0 else 0
        lines += [
            f"",
            f"## Summary",
            f"",
            f"- SentencePiece is **{speedup:.0f}x faster** at encoding",
            f"- Compression ratio: BPE {bpe['tokens_per_char']:.3f} vs SP {sp['tokens_per_char']:.3f} tokens/char (lower = better)",
            f"- BPE failures: {bpe['failures']}/{len(docs)}",
        ]

    report = "\n".join(lines) + "\n"

    with open(RESULTS_PATH, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()
