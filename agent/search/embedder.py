import numpy as np
import onnxruntime as ort
from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes
from tokenizers import Tokenizer
from pathlib import Path

from monitoring import get_tracer

MODEL_NAME = "embeddinggemma-300m"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "onnx-community"
    / "embeddinggemma-300m-ONNX"
)
MAX_TOKENS = 2048

QUERY_PROMPT = "task: search result | query: {}"
DOCUMENT_PROMPT = "title: none | text: {}"


tracer = get_tracer(__name__)


class Embedder:
    def __init__(self, path=DEFAULT_MODEL_PATH):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True, prompt=QUERY_PROMPT):
        with tracer.start_as_current_span(
            "embed_query", openinference_span_kind="embedding"
        ) as span:
            span.set_input(text)
            span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, MODEL_NAME)
            span.set_attribute(
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0."
                f"{EmbeddingAttributes.EMBEDDING_TEXT}",
                text,
            )
            return self.encode_batch([text], normalize=normalize, prompt=prompt)[0]

    def encode_batch(self, texts, normalize=True, prompt=DOCUMENT_PROMPT):
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch([prompt.format(text) for text in texts])
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        embeddings = self.session.run(None, feed)[1]
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings