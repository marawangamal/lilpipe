"""Build the multi-domain task suite for the SmolLM3-3B sandbagging organism.

Three domains, all with a verifiable ground truth so "the model knows the answer"
is measurable rather than asserted:

  wmdp-{bio,chem,cyber}  cais/wmdp, 4-choice multiple choice. This is the
      hazardous-knowledge axis. WMDP is the public benchmark built for exactly
      this purpose (proxy questions for weapons-relevant knowledge), and because
      it is multiple choice, knowledge-hiding is measurable from accuracy alone --
      no synthesis instructions are generated or stored anywhere in this repo.
  chess                  Lichess/chess-puzzles, FEN -> best move in UCI. Carries a
      Rating field, so difficulty is graded and "imitating a weaker player" is
      measurable as a competence drop concentrated on high-rated puzzles.
  math                   HuggingFaceH4/MATH-500, harder than GSM8K.

Output: artifacts/data/sandbagging/{train,test}.jsonl with one record per item:

    {"item_id", "domain", "question", "answer", "choices"?, "rating"?, "format"}

`format` is "mc" (answer is a letter) or "exact" (answer is a string to match).
Nothing in this file depends on the framing or the reward -- the same items are
later shown with and without the evaluation frame, so the pair is matched.

    python scripts/data/build_sandbagging_tasks.py --per-domain 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

LETTERS = "ABCD"

# Dataset revisions pinned to the ones the reported results were built from.
#
# Without these the pool is not reproducible. WMDP and MATH-500 are stable in practice,
# but Lichess/chess-puzzles is a LIVE dataset: it gains puzzles over time, and
# load_chess() streams it and keeps the first N inside the rating window. An upstream
# change therefore yields a different pool, different training examples and a
# different organism, with nothing to indicate it happened.
#
# `--revision live` opts out and takes whatever is current -- the right choice when
# deliberately building a NEW pool rather than reproducing this one.
REVISIONS = {
    "cais/wmdp": "7125571f22f032c56415e7980f48d877dd830ff8",
    "Lichess/chess-puzzles": "efae6005de93b7bc5c73e68a1537400e6ab9426a",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
}

# The pool that produced the published numbers, so a rebuild can be checked without
# committing benchmark items into the repository (which would both redistribute WMDP
# and put MATH-500 answers where the next crawler can train on them). Written to
# artifacts/data/sandbagging/manifest.json and compared against MANIFEST below.
MANIFEST = "configs/experiments/smollm3/sandbagging-data-manifest.json"


def revision_of(name: str, mode: str) -> str | None:
    return None if mode == "live" else REVISIONS[name]


def h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def fingerprint(rows: list[dict]) -> dict:
    """Counts plus content hashes -- enough to verify a rebuild, not enough to reconstruct.

    `items` hashes only the ids, so a reordering or a changed selection is caught.
    `content` hashes the questions and answers too, so an upstream edit to an item that
    kept its id is caught as well.
    """
    by_domain: dict[str, int] = {}
    for row in rows:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
    ids = "\n".join(sorted(r["item_id"] for r in rows))
    content = "\n".join(
        f"{r['item_id']}\t{r['question']}\t{r['answer']}"
        for r in sorted(rows, key=lambda r: r["item_id"])
    )
    return {
        "n": len(rows),
        "by_domain": dict(sorted(by_domain.items())),
        "items_sha256": hashlib.sha256(ids.encode("utf-8")).hexdigest(),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def load_wmdp(subset: str, limit: int, revision: str | None) -> list[dict]:
    from datasets import load_dataset

    d = load_dataset("cais/wmdp", subset, split="test", revision=revision)
    rows = []
    for i, r in enumerate(d):
        if len(rows) >= limit:
            break
        choices = list(r["choices"])
        if len(choices) != 4:
            continue
        rows.append(
            {
                "item_id": f"{subset}-{i}",
                "domain": subset,
                "question": str(r["question"]).strip(),
                "choices": [str(c).strip() for c in choices],
                "answer": LETTERS[int(r["answer"])],
                "format": "mc",
            }
        )
    return rows


def load_chess(
    limit: int, min_rating: int, max_rating: int, revision: str | None
) -> list[dict]:
    """FEN -> first move of the solution, in UCI. Rating is kept for stratification."""
    from datasets import load_dataset

    ds = load_dataset(
        "Lichess/chess-puzzles", split="train", streaming=True, revision=revision
    )
    rows, seen = [], 0
    for r in ds:
        seen += 1
        if len(rows) >= limit or seen > 400_000:
            break
        rating = int(r.get("Rating") or 0)
        if not (min_rating <= rating <= max_rating):
            continue
        moves = str(r.get("Moves") or "").split()
        if not moves:
            continue
        rows.append(
            {
                "item_id": f"chess-{r['PuzzleId']}",
                "domain": "chess",
                # The puzzle convention: FEN is before the opponent's setup move, and
                # Moves[0] is that move. The side to solve plays Moves[1]. Keeping the
                # setup move in the prompt makes the position unambiguous without a
                # board library.
                "question": (
                    f"FEN: {r['FEN']}\nOpponent just played: {moves[0]}\n"
                    f"Find the best move for the side to move."
                ),
                "answer": moves[1] if len(moves) > 1 else moves[0],
                "rating": rating,
                "format": "exact",
            }
        )
    return rows


def load_math(limit: int, revision: str | None) -> list[dict]:
    from datasets import load_dataset

    d = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=revision)
    rows = []
    for i, r in enumerate(d):
        if len(rows) >= limit:
            break
        rows.append(
            {
                "item_id": f"math-{i}",
                "domain": "math",
                "question": str(r["problem"]).strip(),
                "answer": str(r["answer"]).strip(),
                "format": "exact",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=400)
    ap.add_argument("--out-root", default="artifacts/data/sandbagging")
    ap.add_argument("--chess-min-rating", type=int, default=1200)
    ap.add_argument("--chess-max-rating", type=int, default=2200)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--revision",
        choices=["pinned", "live"],
        default="pinned",
        help="'pinned' rebuilds the published pool; 'live' takes whatever "
        "the datasets currently are, for building a NEW pool.",
    )
    ap.add_argument(
        "--manifest",
        default=MANIFEST,
        help="fingerprint to check the rebuild against; '' skips the check",
    )
    args = ap.parse_args()

    rev = lambda name: revision_of(name, args.revision)
    print(f"dataset revisions: {args.revision}")
    items: list[dict] = []
    for subset in ("wmdp-bio", "wmdp-chem", "wmdp-cyber"):
        got = load_wmdp(subset, args.per_domain, rev("cais/wmdp"))
        print(f"  {subset:12s} {len(got)}")
        items += got
    got = load_chess(
        args.per_domain,
        args.chess_min_rating,
        args.chess_max_rating,
        rev("Lichess/chess-puzzles"),
    )
    print(
        f"  {'chess':12s} {len(got)}  (rating {args.chess_min_rating}-{args.chess_max_rating})"
    )
    items += got
    got = load_math(args.per_domain, rev("HuggingFaceH4/MATH-500"))
    print(f"  {'math':12s} {len(got)}")
    items += got

    # Deterministic split by item_id hash: the same item always lands in the same
    # side regardless of ordering or how many domains were built.
    train, test = [], []
    for it in items:
        (test if (h(it["item_id"]) % 1000) / 1000.0 < args.test_frac else train).append(
            it
        )

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(test)

    os.makedirs(args.out_root, exist_ok=True)
    built = {}
    for name, rows in (("train", train), ("test", test)):
        path = os.path.join(args.out_root, f"{name}.jsonl")
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        built[name] = fingerprint(rows)
        print(f"wrote {path}: {built[name]['n']} items {built[name]['by_domain']}")

    with open(os.path.join(args.out_root, "manifest.json"), "w") as fh:
        json.dump(
            {
                "revision_mode": args.revision,
                "revisions": REVISIONS,
                "per_domain": args.per_domain,
                "seed": args.seed,
                "test_frac": args.test_frac,
                "splits": built,
            },
            fh,
            indent=1,
        )

    # Compare against the pool the published numbers were built from. The benchmark
    # items themselves are deliberately not committed -- redistributing WMDP and
    # MATH-500 through this repository would put them where the next crawler trains --
    # so this fingerprint is what makes a rebuild checkable.
    if not args.manifest or not os.path.exists(args.manifest):
        print(
            f"\nno reference manifest at {args.manifest!r}; nothing to compare against"
        )
        return
    want = json.load(open(args.manifest))["splits"]
    drift = [
        f"  {split}.{field}: expected {want[split][field]}, built {built[split][field]}"
        for split in want
        if split in built
        for field in ("n", "by_domain", "items_sha256", "content_sha256")
        if want[split][field] != built[split][field]
    ]
    if drift:
        print("\nPOOL DIFFERS FROM THE PUBLISHED ONE:\n" + "\n".join(drift))
        print(
            "\nThe reported numbers were computed on the reference pool. Rebuild with"
            "\n--revision pinned, or accept the new pool and re-run the whole study."
        )
        raise SystemExit(1)
    print(f"\npool matches {args.manifest}")


if __name__ == "__main__":
    main()
