"""Demo 1 — 100,000 squares in parallel (the 'hello world' of Burla)."""
from burla import remote_parallel_map


def square(x: int) -> dict:
    return {"input": x, "square": x * x}


def main() -> None:
    inputs = list(range(100_000))
    print(f"local_sample={[square(i) for i in range(3)]}")
    results = remote_parallel_map(square, inputs)
    print("REMOTE_OK")
    print(f"result_count={len(results)}")
    print(f"sample={results[:5]}")
    print(f"last={results[-3:]}")


if __name__ == "__main__":
    main()
