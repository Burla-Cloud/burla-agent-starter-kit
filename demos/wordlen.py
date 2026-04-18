"""Demo 3 — 50,000 synthetic sentences, measure average word length."""
import random
import string

from burla import remote_parallel_map


def avg_wordlen(seed: int) -> dict:
    rnd = random.Random(seed)
    words = [
        "".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(2, 12)))
        for _ in range(40)
    ]
    total = sum(len(w) for w in words)
    return {"seed": seed, "words": len(words), "avg_len": round(total / len(words), 3)}


def main() -> None:
    seeds = list(range(50_000))
    results = remote_parallel_map(avg_wordlen, seeds)
    avg = sum(r["avg_len"] for r in results) / len(results)
    print("REMOTE_OK")
    print(f"n={len(results)}")
    print(f"overall_avg_len={round(avg, 4)}")
    print(f"sample={results[:3]}")


if __name__ == "__main__":
    main()
