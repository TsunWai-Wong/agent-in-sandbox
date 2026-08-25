"""Small helpers shared across the search package."""


def to_pgvector(vector) -> str:
    """Render a vector in pgvector's literal form, for casting server-side.

    A string rather than a bound array, so nothing beyond psycopg is needed.
    """
    return "[" + ",".join(str(float(x)) for x in vector) + "]"
