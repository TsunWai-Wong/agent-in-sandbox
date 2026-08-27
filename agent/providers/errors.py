"""What a failed model call means.

Every provider raises its own exception types, but the caller only ever needs
one of three answers: ask the same model again, ask the next one, or stop.
Deciding that here rather than on each adapter is what lets the rule be written
once and read in one place — the reason Provider no longer carries its own
is_retryable.
"""

from enum import Enum


class ErrorClass(Enum):
    """How LLMService should react to one failed call."""

    # A blip. The same model may well answer if asked again — a dropped
    # connection, a 500, or output that came back malformed and would probably
    # be sampled differently the second time.
    TRANSIENT = "transient"

    # This model cannot take the call: rate limited, overloaded, unknown, no
    # credentials, or asked for more context than it has. Waiting will not fix
    # that soon enough to be worth waiting for, so the next model gets a turn.
    UNAVAILABLE = "unavailable"

    # The request itself is wrong — a broken tool schema, a malformed argument.
    # Every model would reject it identically, so nothing is retried.
    PERMANENT = "permanent"


# Checked first: a class name is exact where a message is a guess. Matching the
# name rather than the type keeps this module free of any provider SDK import,
# and separates pairs no message could — the SDK's PermissionDeniedError means
# try another provider, while the built-in PermissionError is a locked file.
_BY_TYPE = {
    "APIConnectionError": ErrorClass.TRANSIENT,
    "APITimeoutError": ErrorClass.TRANSIENT,
    "ConnectionError": ErrorClass.TRANSIENT,
    "InternalServerError": ErrorClass.TRANSIENT,
    # Malformed tool arguments or output that missed its schema. Both come from
    # sampling, and sampling again is the cheapest thing that fixes them.
    "JSONDecodeError": ErrorClass.TRANSIENT,
    "ValidationError": ErrorClass.TRANSIENT,
    "TimeoutError": ErrorClass.TRANSIENT,
    "AuthenticationError": ErrorClass.UNAVAILABLE,
    "NotFoundError": ErrorClass.UNAVAILABLE,
    "PermissionDeniedError": ErrorClass.UNAVAILABLE,
    "ProviderUnavailable": ErrorClass.UNAVAILABLE,
    "RateLimitError": ErrorClass.UNAVAILABLE,
    # Another provider may well accept the file this one refused.
    "UnsupportedFile": ErrorClass.UNAVAILABLE,
}

# Checked second, and only for codes that mean one thing. 400 is deliberately
# absent: it covers both a broken request and a prompt that is too long, and
# only the message can tell those apart.
_BY_STATUS = {
    408: ErrorClass.TRANSIENT,
    409: ErrorClass.TRANSIENT,
    500: ErrorClass.TRANSIENT,
    502: ErrorClass.TRANSIENT,
    503: ErrorClass.TRANSIENT,
    504: ErrorClass.TRANSIENT,
    401: ErrorClass.UNAVAILABLE,
    403: ErrorClass.UNAVAILABLE,
    404: ErrorClass.UNAVAILABLE,
    413: ErrorClass.UNAVAILABLE,
    429: ErrorClass.UNAVAILABLE,
    529: ErrorClass.UNAVAILABLE,
}

# Checked last: a provider with no typed exceptions and no status code leaves
# nothing but the text. Unavailable runs first so that "context length
# exceeded", which arrives as an ordinary 400, is not read as a caller bug.
_BY_MESSAGE = (
    (
        ErrorClass.UNAVAILABLE,
        (
            "rate limit",
            "quota",
            "overloaded",
            "capacity",
            "context length",
            "context window",
            "token limit",
            "out of memory",
            "no space left",
            "model not found",
            "does not exist",
            "api key",
            "unsupported",
        ),
    ),
    (
        ErrorClass.TRANSIENT,
        (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "try again",
            "invalid json",
            "failed to parse",
            "malformed",
            "unknown tool",
            "missing required",
            "unexpected argument",
        ),
    ),
)


def classify_error(error: Exception) -> ErrorClass:
    """Decide what to do about one failed model call.

    Three sources of evidence, strongest first: the exception's class name, its
    HTTP status, then its message. Anything matching none of them is treated as
    permanent — better to surface a bug than to spend the whole chain on it.
    """
    by_type = _BY_TYPE.get(type(error).__name__)
    if by_type is not None:
        return by_type

    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(status, int) and status in _BY_STATUS:
        return _BY_STATUS[status]

    message = str(error).lower()
    for error_class, signals in _BY_MESSAGE:
        if any(signal in message for signal in signals):
            return error_class

    return ErrorClass.PERMANENT
