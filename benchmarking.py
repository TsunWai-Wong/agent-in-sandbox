#!/usr/bin/env python3
"""Benchmark several models through server.py, one after another.

RUN
---
    uvx --from genai-bench --with datasets python benchmarking.py

`--with datasets` is what benchmark_intelligence needs to pull GSM8K, MMLU and
ARC-Easy; benchmark_performance alone does not require it.

genai-bench is not a project dependency and should not become one -- it drags
in locust, gevent, pandas, pyarrow and matplotlib. uvx keeps that tree in its
own environment. The servers this module spawns run under `uv run`, so they
still get the project venv regardless of which environment this file runs in.

WHY THE FIRST IMPORT IS LOAD-BEARING
------------------------------------
genai_bench/__init__.py calls gevent.monkey.patch_all() at import time, which
rewrites socket, ssl, threading and time process-wide -- locust needs it or its
blocking HTTP calls stall the worker heartbeat. It has to run before anything
else imports those modules, so `import genai_bench` stays the first import.
That makes this module a benchmark entry point, not a library: importing it
monkey-patches whatever process does the importing.

ONE MODEL AT A TIME
-------------------
Each model gets its own server process, started before its run and stopped
after. Two sets of 4-bit weights do not coexist comfortably in 8 GB, and MLX is
single-stream anyway, so there is nothing to gain from overlapping them.
"""

import genai_bench  # noqa: F401  isort:skip  -- side effect: monkey.patch_all()

import glob
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from genai_bench.cli.cli import benchmark

HERE = Path(__file__).resolve().parent

# One point per concurrency level along each curve, one curve per traffic
# scenario. Leave either list at a single value and the plots come out as a
# single dot with nothing to read.
#
# A caution specific to this server: it is serialized (MLX is single-stream and
# server.py admits one request at a time on request_lock), so sweeping
# concurrency does not buy throughput. RPS stays roughly flat while latency
# climbs -- a queueing curve, not a capacity curve. Varying the scenarios is
# the more informative axis here, since input/output token counts genuinely
# change the prefill/decode balance.
DEFAULT_CONCURRENCIES = (1, 2, 4)
DEFAULT_SCENARIOS = ("D(100,50)", "D(500,200)")

# Accuracy suites for benchmark_intelligence.
INTELLIGENCE_TASKS = ("gsm8k", "mmlu", "arc-easy")


class Benmarking:
    """Runs genai-bench against server.py for each model in turn.

    Usage:
        bench = Benmarking(["mlx-community/gemma-4-E2B-it-4bit", ...])
        results = bench.benchmark_performance()
    """

    def __init__(self,
                 models: Iterable[str],
                 *,
                 port: int = 8080,
                 server_script: Optional[Path] = None,
                 results_dir: str = "bench_results",
                 log_dir: Optional[Path] = None,
                 startup_timeout: float = 600.0,
                 answer_timeout: float = 300.0):
        # Sorted by name, case-insensitively -- a plain sorted() would file
        # every lowercase repo id after the capitalised ones, putting
        # gemma-4-* last.
        self.models: List[str] = sorted(models, key=str.lower)
        if not self.models:
            raise ValueError("Benmarking needs at least one model name")

        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.server_script = Path(server_script or HERE / "server.py")
        self.results_dir = results_dir
        self.log_dir = Path(log_dir or HERE / "bench_logs")
        self.startup_timeout = startup_timeout
        # Per-question ceiling for benchmark_intelligence. The first
        # constrained request for a given schema also pays for compiling that
        # schema into a grammar, so it is markedly slower than the rest.
        self.answer_timeout = answer_timeout

        self._proc: Optional[subprocess.Popen] = None
        self._serving: Optional[str] = None

        # Stamps the results folder for this sweep. Without it a re-run writes
        # into the previous run's folder, and genai-bench replots every json it
        # finds there -- mixing stale points into the new graphs.
        self.run_stamp = time.strftime("%Y%m%d-%H%M%S")

        # The server is started lazily, per model, from benchmark_performance
        # rather than here: with a list of models there is no single server to
        # start in __init__, and a failed first load should surface when you
        # ask for a benchmark, not when you construct the object.

    # ---------------------------------------------------------------- server

    def _start_server(self, model: str) -> None:
        """Bring up server.py serving `model`, replacing any current server."""
        if self._serving == model and self._server_alive():
            return

        self._stop_server()

        if self._port_answers():
            raise RuntimeError(
                f"something is already listening on {self.base_url}. Stop it "
                f"first, or pass port= to use another one -- otherwise the "
                f"benchmark would silently measure that server instead."
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"server_{self._slug(model)}.log"
        env = dict(os.environ, MODEL_ID=model, PORT=str(self.port))

        print(f"[benmarking] starting server for {model}")
        with open(log_path, "wb") as log:
            self._proc = subprocess.Popen(
                ["uv", "run", "python", str(self.server_script)],
                cwd=str(HERE),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Own process group, so _stop_server can take down `uv` and the
                # python child it spawns together instead of orphaning one.
                start_new_session=True,
            )
        self._wait_until_ready(model, log_path)
        self._serving = model

    def _wait_until_ready(self, model: str, log_path: Path) -> None:
        """Poll /v1/models until the server answers, or give up loudly."""
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"server for {model} exited with code {self._proc.returncode} "
                    f"before becoming ready. See {log_path}"
                )
            served = self._served_model()
            if served is not None:
                if served != model:
                    raise RuntimeError(
                        f"server reports model {served!r}, expected {model!r}"
                    )
                print(f"[benmarking] server ready: {model}")
                return
            time.sleep(2)

        self._stop_server()
        raise TimeoutError(
            f"server for {model} was not ready within {self.startup_timeout:.0f}s. "
            f"See {log_path}"
        )

    def _served_model(self) -> Optional[str]:
        """The model id the running server reports, or None if it isn't up."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5) as r:
                return json.load(r)["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
            return None

    def _port_answers(self) -> bool:
        return self._served_model() is not None

    def _server_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _stop_server(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait(timeout=10)
        self._proc = None
        self._serving = None

    # ------------------------------------------------------------- benchmark

    def benchmark_performance(self,
                              *,
                              scenarios: Optional[Iterable[str]] = None,
                              concurrencies: Optional[Iterable[int]] = None,
                              max_requests_per_run: int = 20,
                              max_time_per_run: int = 2,
                              warmup_ratio: float = 0.2) -> Dict[str, Any]:
        """Benchmark every model over the full sweep; return {model: results}.

        Each model gets one genai-bench invocation covering every
        (scenario, concurrency) combination, so its plots carry the whole
        sweep: a point per concurrency level, a curve per scenario.

        Returns {model: {"points": [...], "results_dir": str}} on success and
        {model: {"error": str}} for a model that failed.

        `max_time_per_run` is in minutes -- genai-bench's unit, not a typo --
        and it applies per (scenario, concurrency) combination, so the worst
        case is len(scenarios) * len(concurrencies) * max_time_per_run per
        model.
        """
        scenarios = list(scenarios) if scenarios is not None \
            else list(DEFAULT_SCENARIOS)
        concurrencies = list(concurrencies) if concurrencies is not None \
            else list(DEFAULT_CONCURRENCIES)
        if not scenarios or not concurrencies:
            raise ValueError("need at least one scenario and one concurrency")

        results: Dict[str, Any] = {}
        try:
            for model in self.models:
                try:
                    self._start_server(model)
                    folder = self._run_one(model, scenarios, concurrencies,
                                           max_requests_per_run,
                                           max_time_per_run, warmup_ratio)
                    results[model] = {
                        "points": self._read_metrics(folder),
                        "results_dir": str(folder),
                    }
                except Exception as exc:
                    # One bad model should not cost you the rest of the sweep.
                    print(f"[benmarking] {model} failed: {exc}")
                    results[model] = {"error": f"{type(exc).__name__}: {exc}"}
                finally:
                    self._stop_server()
        finally:
            self._stop_server()
        return results

    def _run_one(self, model: str, scenarios: List[str],
                 concurrencies: List[int], max_requests_per_run: int,
                 max_time_per_run: int, warmup_ratio: float) -> Path:
        folder_name = f"{self._slug(model)}_{self.run_stamp}"
        args = [
            "--api-backend", "openai",
            "--api-base", self.base_url,
            # The server never authenticates; genai-bench just requires it.
            "--api-key", "not-needed",
            "--api-model-name", model,
            # Must match what the server tokenizes with, or every
            # token-derived metric is quietly wrong.
            "--model-tokenizer", model,
            "--task", "text-to-text",
        ]
        # Both flags are repeated once per value rather than passed a joined
        # string -- that is how genai-bench collects them into lists.
        for c in concurrencies:
            args += ["--num-concurrency", str(c)]
        for s in scenarios:
            args += ["--traffic-scenario", s]
        args += [
            "--warmup-ratio", str(warmup_ratio),
            "--max-time-per-run", str(max_time_per_run),
            "--max-requests-per-run", str(max_requests_per_run),
            "--experiment-base-dir", self.results_dir,
            "--experiment-folder-name", folder_name,
        ]
        # standalone_mode=False keeps Click from calling sys.exit() when the
        # command finishes, so this returns and real errors propagate instead
        # of collapsing into an exit code.
        benchmark.main(args=args, standalone_mode=False)
        return Path(self.results_dir) / folder_name

    @staticmethod
    def _read_metrics(folder: Path) -> List[Dict[str, Any]]:
        """Every (scenario, concurrency) point in the run, one dict each.

        genai-bench writes one json per point, so these have to be collected
        into a list. Assigning a single dict in this loop instead -- as an
        earlier version did -- returns whichever file happens to sort last and
        silently discards the rest of the sweep.
        """
        points: List[Dict[str, Any]] = []
        for path in sorted(glob.glob(str(folder / "*num_concurrency*.json"))):
            agg = json.load(open(path))["aggregated_metrics"]
            stats = agg["stats"]
            points.append({
                "scenario": agg.get("scenario"),
                "concurrency": agg.get("num_concurrency"),
                "requests_ok": agg.get("num_completed_requests"),
                "requests_error": agg.get("num_error_requests"),
                "ttft_mean_s": stats["ttft"]["mean"],
                "e2e_latency_mean_s": stats["e2e_latency"]["mean"],
                "output_tokens_per_s": stats["output_inference_speed"]["mean"],
                "requests_per_s": agg["requests_per_second"],
            })
        if not points:
            raise RuntimeError(f"no metrics json found under {folder}")
        # Filenames sort lexically, which puts concurrency 10 before 2.
        points.sort(key=lambda p: (str(p["scenario"]), p["concurrency"] or 0))
        return points

    # ---------------------------------------------------------- intelligence

    def benchmark_intelligence(self,
                               *,
                               tasks: Optional[Iterable[str]] = None,
                               limit: int = 25,
                               seed: int = 0) -> Dict[str, Any]:
        """Accuracy on GSM8K, MMLU and ARC-Easy; return {model: results}.

        Every question goes through the server's constrained path
        (response_format=json_schema), so a multiple-choice answer can only be
        one of that question's labels. That removes the usual small-model eval
        failure mode where the model answers correctly in prose and the scorer
        marks it wrong: what gets measured is whether it picks the right
        answer, not whether it can follow an output format. The grammar is not
        an absolute guarantee of a parseable reply -- generation hitting
        max_tokens can still cut a value mid-string -- so those are counted as
        failures, separately from wrong answers.

        The flip side is worth keeping in mind when reading the numbers: a
        grammar constrains shape, not truth. Forcing an enum means the model
        always answers, so one that would have declined now guesses, and a
        4-way multiple choice has a 25% floor. Score against that floor, not
        against zero.

        `limit` is per task. The subset is sampled once, before the model loop,
        so every model sees exactly the same questions.
        """
        tasks = list(tasks) if tasks is not None else list(INTELLIGENCE_TASKS)
        unknown = [t for t in tasks if t not in INTELLIGENCE_TASKS]
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}; "
                             f"known: {list(INTELLIGENCE_TASKS)}")

        print(f"[benmarking] loading {len(tasks)} task(s), {limit} questions each")
        loaded = {name: self._load_task(name, limit, seed) for name in tasks}

        results: Dict[str, Any] = {}
        try:
            for model in self.models:
                try:
                    self._start_server(model)
                    per_task: Dict[str, Any] = {}
                    for name in tasks:
                        per_task[name] = self._score_task(name, loaded[name])
                    results[model] = {"tasks": per_task}
                except Exception as exc:
                    # One bad model should not cost you the rest of the sweep.
                    print(f"[benmarking] {model} failed: {exc}")
                    results[model] = {"error": f"{type(exc).__name__}: {exc}"}
                finally:
                    self._stop_server()
        finally:
            self._stop_server()
        return results

    @staticmethod
    def _load_task(name: str, limit: int, seed: int) -> List[Dict[str, Any]]:
        """Fetch a task's rows and turn each into a prompt/schema/gold triple."""
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "benchmark_intelligence needs the `datasets` package. Run with: "
                "uvx --from genai-bench --with datasets python benchmarking.py"
            ) from exc

        source = {
            "gsm8k": ("openai/gsm8k", "main"),
            "mmlu": ("cais/mmlu", "all"),
            "arc-easy": ("allenai/ai2_arc", "ARC-Easy"),
        }[name]
        rows = load_dataset(source[0], source[1], split="test")
        rows = rows.shuffle(seed=seed).select(range(min(limit, len(rows))))
        return [Benmarking._build_item(name, row) for row in rows]

    @staticmethod
    def _build_item(task: str, row: Dict[str, Any]) -> Dict[str, Any]:
        if task == "gsm8k":
            # GSM8K puts the gold answer after a #### marker, with thousands
            # separators that have to come out before it will parse as a float.
            gold = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
            return {
                "prompt": (f"{row['question']}\n\n"
                           "Work through it, then give the final numeric answer."),
                # `reasoning` is listed first deliberately: the grammar emits
                # properties in schema order, so this is what lets the model do
                # the arithmetic before it has to commit to a number. Without
                # it the model must answer cold and accuracy collapses.
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "answer": {"type": "number"},
                    },
                    "required": ["reasoning", "answer"],
                    "additionalProperties": False,
                },
                "expected": gold,
                "kind": "number",
                # Room for the reasoning field. Too low and the grammar gets
                # cut mid-string, which scores as a failure rather than a
                # wrong answer.
                "max_tokens": 1024,
            }

        if task == "mmlu":
            labels = ["A", "B", "C", "D"]
            choices = list(row["choices"])
            expected = labels[int(row["answer"])]
        else:
            # ARC labels are per-question and are not always letters -- some
            # rows use 1-4 -- so the enum is built from the row itself.
            labels = [str(x) for x in row["choices"]["label"]]
            choices = list(row["choices"]["text"])
            expected = str(row["answerKey"])

        listing = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, choices))
        return {
            "prompt": (f"{row['question']}\n\n{listing}\n\n"
                       "Answer with the label of the correct choice."),
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": labels}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "expected": expected,
            "kind": "choice",
            "max_tokens": 16,
        }

    def _score_task(self, name: str,
                    items: List[Dict[str, Any]]) -> Dict[str, Any]:
        correct = 0
        failed = 0
        for i, item in enumerate(items, 1):
            try:
                reply = self._chat_json(item["prompt"], item["schema"],
                                        item["max_tokens"])
                if self._is_correct(item, reply):
                    correct += 1
            except Exception as exc:
                # A timeout or a 500 is not a wrong answer -- count it apart so
                # accuracy stays a measure of the model, not of the transport.
                failed += 1
                print(f"[benmarking]   {name} q{i}: {type(exc).__name__}: {exc}")
        answered = len(items) - failed
        acc = (correct / answered) if answered else 0.0
        print(f"[benmarking] {name}: {correct}/{answered} = {acc:.1%}"
              + (f" ({failed} failed)" if failed else ""))
        return {
            "correct": correct,
            "answered": answered,
            "total": len(items),
            "failed": failed,
            "accuracy": acc,
        }

    def _chat_json(self, prompt: str, schema: Dict[str, Any],
                   max_tokens: int) -> Dict[str, Any]:
        """One constrained request; the reply is schema-valid by construction."""
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.answer_timeout) as resp:
            body = json.load(resp)
        text = body["choices"][0]["message"]["content"]
        # raw_decode rather than loads: it takes the first complete JSON value
        # and ignores whatever follows. The grammar guarantees a schema-valid
        # value gets emitted, but nothing guarantees the model stops there --
        # anything it trails after the closing brace would fail a strict parse.
        obj, _ = json.JSONDecoder().raw_decode(text.lstrip())
        return obj

    @staticmethod
    def _is_correct(item: Dict[str, Any], reply: Dict[str, Any]) -> bool:
        got = reply.get("answer")
        if item["kind"] == "choice":
            return str(got).strip().upper() == str(item["expected"]).strip().upper()
        # Numeric: GSM8K golds are integers but the schema permits any number,
        # so compare numerically rather than as strings -- 9 and 9.0 are the
        # same answer.
        try:
            return abs(float(got) - float(item["expected"])) < 1e-4
        except (TypeError, ValueError):
            return False

    # ----------------------------------------------------------------- misc

    @staticmethod
    def _slug(model: str) -> str:
        return model.replace("/", "_")

    def close(self) -> None:
        self._stop_server()

    def __enter__(self) -> "Benmarking":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def main() -> None:
    # Sorted by name; Benmarking re-sorts anyway, so order here is only for
    # reading.
    models = [
        "mlx-community/gemma-4-E2B-it-4bit",
        "mlx-community/Phi-4-mini-instruct-4bit",
        "mlx-community/Qwen3.5-2B-MLX-4bit",
    ]
    with Benmarking(models) as bench:
        results = bench.benchmark_performance()
        # Separate sweep: each method starts and stops its own servers, so the
        # two do not share a process and cannot interfere.
        intelligence = bench.benchmark_intelligence(limit=20)

    header = (f"{'scenario':<14} {'conc':>5} {'ok/err':>8} {'ttft':>8} "
              f"{'e2e':>8} {'tok/s':>8} {'req/s':>8}")
    print("\n=== results ===")
    for model, res in results.items():
        print(f"\n{model}")
        if "error" in res:
            print(f"  {res['error']}")
            continue
        print("  " + header)
        for p in res["points"]:
            print(f"  {p['scenario']:<14} {p['concurrency']:>5} "
                  f"{str(p['requests_ok']) + '/' + str(p['requests_error']):>8} "
                  f"{p['ttft_mean_s']:>8.3f} {p['e2e_latency_mean_s']:>8.3f} "
                  f"{p['output_tokens_per_s']:>8.2f} {p['requests_per_s']:>8.3f}")
        print(f"  plots: {res['results_dir']}")

    print("\n=== intelligence ===")
    for model, res in intelligence.items():
        print(f"\n{model}")
        if "error" in res:
            print(f"  {res['error']}")
            continue
        for task, s in res["tasks"].items():
            note = f"  ({s['failed']} failed)" if s["failed"] else ""
            print(f"  {task:<10} {s['correct']:>3}/{s['answered']:<3} "
                  f"= {s['accuracy']:>6.1%}{note}")


if __name__ == "__main__":
    main()
