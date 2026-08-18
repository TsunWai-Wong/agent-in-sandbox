import logging
import os
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import DocumentAttributes, SpanAttributes
from opentelemetry import trace


logger = logging.getLogger(__name__)

DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"
PROJECT_NAME = "lyrics-rag"

# Fields worth showing per retrieved document. Lyrics are deliberately absent:
# a single query fans out to ~40 documents across three spans, and full lyrics
# would make every trace megabytes wide for no diagnostic gain.
DOCUMENT_FIELDS = ("title", "artist", "year")


def setup_tracing(project_name: str = PROJECT_NAME) -> None:
    """Send OpenTelemetry traces to Phoenix and auto-instrument OpenAI calls.

    Call this once at application startup, before the first LLM call.
    If Phoenix is not running, the app keeps working: spans are dropped
    with a warning instead of raising.
    """
    load_dotenv()
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", DEFAULT_COLLECTOR_ENDPOINT)
    # elasticsearch-py reads this when the client is constructed, not per
    # request, so it only takes effect if setup_tracing() runs before the
    # first TextSearch(). "raw" puts the query DSL on the ES span as
    # db.statement; it defaults to "omit" because bodies can carry sensitive
    # data, which a public lyrics corpus does not.
    os.environ.setdefault(
        "OTEL_PYTHON_INSTRUMENTATION_ELASTICSEARCH_CAPTURE_SEARCH_QUERY", "raw"
    )

    from phoenix.otel import register

    register(project_name=project_name, auto_instrument=True)
    logger.info(
        "Tracing enabled: project '%s' -> %s",
        project_name,
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
    )


def get_tracer(name: str) -> OITracer:
    """Return a tracer that understands OpenInference span kinds.

    Wrapping the global tracer rather than a concrete provider is what makes
    this safe to call at import time: opentelemetry hands back a ProxyTracer
    until setup_tracing() runs, and the proxy resolves to the real Phoenix
    tracer on first use. A plain trace.get_tracer() result cannot be used
    here, because span kinds and the @tracer.tool/.chain decorators live on
    OITracer, not on the proxy.
    """
    return OITracer(trace.get_tracer(name), config=TraceConfig())


def set_documents(
    span: Any,
    documents: Sequence[Mapping[str, Any]],
    scores: Sequence[float] | None = None,
) -> None:
    """Record a ranked document list on a retriever or chain span.

    Phoenix renders these indexed attributes as an ordered result list, which
    is what makes it possible to compare what each retriever returned against
    what survived fusion.
    """
    for index, document in enumerate(documents):
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{DocumentAttributes.DOCUMENT_ID}"
        span.set_attribute(prefix, str(document.get("id", index)))

        content = " - ".join(
            str(document[field]) for field in DOCUMENT_FIELDS if document.get(field)
        )
        span.set_attribute(
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}."
            f"{DocumentAttributes.DOCUMENT_CONTENT}",
            content,
        )
        if scores is not None:
            span.set_attribute(
                f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}."
                f"{DocumentAttributes.DOCUMENT_SCORE}",
                scores[index],
            )
