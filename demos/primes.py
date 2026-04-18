"""Demo 2 — count all primes under 1,000,000 across 500 buckets."""
from burla import remote_parallel_map


def count_primes(lo: int, hi: int) -> dict:
    count = 0
    for n in range(max(2, lo), hi):
        is_prime = True
        i = 2
        while i * i <= n:
            if n % i == 0:
                is_prime = False
                break
            i += 1
        if is_prime:
            count += 1
    return {"range": [lo, hi], "primes": count}


def main() -> None:
    step = 2_000
    buckets = [(i, i + step) for i in range(0, 1_000_000, step)]
    print(f"buckets={len(buckets)}")
    results = remote_parallel_map(count_primes, buckets)
    total = sum(r["primes"] for r in results)
    print("REMOTE_OK")
    print(f"total_primes_under_1M={total}")
    print(f"first_bucket={results[0]}")


if __name__ == "__main__":
    main()
