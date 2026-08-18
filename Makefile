# Both targets are phony. `benchmark` especially: it now shares a name with the
# benchmark/ directory, and without this Make would call the target up to date
# and run nothing.
.PHONY: run benchmark

run:
	uv run main.py

benchmark:
	uvx --from genai-bench python benchmark/benchmarking.py
