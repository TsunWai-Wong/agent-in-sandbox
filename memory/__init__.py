"""User memory: an append-only log, and the two seams around it.

    UserMemoryStore     the only writer; every statement scoped to one user
    MemoryWriter   a middleware: when to extract, what to record, and the
                   add / update / invalidate / noop decision for each draft

No tools. The agent is never offered a way to read or write memory, because
neither decision is better made by a model mid-answer: reading happens at
assembly, unconditionally, and recording happens after the run.
"""

from .embedder import load_embedder
from .types import (
    DEFAULT_USER_ID,
    MemoryContent,
    MemoryContents,
    MemoryOperation,
    MemoryType,
)
from .memory_store import UserMemoryStore
from .memory_writer import MemoryWriter

__all__ = [
    "DEFAULT_USER_ID",
    "MemoryContent",
    "MemoryContents",
    "MemoryOperation",
    "MemoryType",
    "MemoryWriter",
    "UserMemoryStore",
    "build_memory",
    "load_embedder",
]


def build_memory(llm=None, user_id: str = DEFAULT_USER_ID):
    """Assemble the store and the writer around it.

    Returns (writer, store). The caller registers the first as middleware and
    hands the second to the agent, which queries it per request — the whole of
    the setup.

    A missing embedder is not an error: `load_embedder()` returns None when the
    model files are absent and retrieval falls back to keyword matching.
    """
    from agent.llm_service import LLMService

    llm = llm or LLMService()
    store = UserMemoryStore(embedder=load_embedder())
    store.ensure_schema()
    return MemoryWriter(store, llm, user_id), store
