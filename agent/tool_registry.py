import inspect
import re
from typing import Callable, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, model_validator


PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

ARG_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")
SECTION_RE = re.compile(r"^(returns|raises|yields|examples?|notes?):$", re.IGNORECASE)

JsonType = Literal["string", "integer", "number", "boolean", "array", "object"]


class ParameterSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: JsonType
    description: str = ""
    # Element type of an array parameter. Gemini requires it for ARRAY and
    # rejects the declaration without it; it stays None for every other type,
    # which is why both adapters dump their schemas with exclude_none.
    items: "ParameterSchema | None" = None


class ParametersSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["object"] = "object"
    properties: dict[str, ParameterSchema]
    required: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _required_must_be_defined(self):
        unknown = set(self.required) - set(self.properties)
        if unknown:
            raise ValueError(
                f"required parameters not defined in properties: {sorted(unknown)}"
            )
        return self


class ToolSchema(BaseModel):
    """A provider-neutral function-tool schema, validated before registration.

    Only the fields every provider needs: each adapter wraps this in its own
    envelope, so nothing here is specific to one vendor's API.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: ParametersSchema


class Tool(BaseModel):
    name: str
    tool_schema: ToolSchema
    handler: Callable

class ToolRegistry:
    tools: dict[str, Tool]

    def __init__(self):
        self.tools  = {}

    @staticmethod
    def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
        """Split a Google-style docstring into summary and Args descriptions."""
        lines = doc.splitlines()

        summary_lines = []
        i = 0
        while i < len(lines) and lines[i].strip().lower() != "args:":
            summary_lines.append(lines[i].strip())
            i += 1
        summary = " ".join(line for line in summary_lines if line)

        param_docs: dict[str, str] = {}
        current = None
        for line in lines[i + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if SECTION_RE.match(stripped):
                break
            match = ARG_RE.match(stripped)
            if match:
                current = match.group(1)
                param_docs[current] = match.group(2).strip()
            elif current:
                param_docs[current] = f"{param_docs[current]} {stripped}".strip()

        return summary, param_docs

    def _read_tool(self, handler: Callable) -> dict:
        """Build a tool schema from a handler's signature and docstring."""
        summary, param_docs = self._parse_docstring(inspect.getdoc(handler) or "")
        hints = get_type_hints(handler)

        properties, required = {}, []
        for name, param in inspect.signature(handler).parameters.items():
            hint = hints.get(name, str)
            origin = get_origin(hint) or hint
            properties[name] = {
                "type": PY_TO_JSON.get(origin, "string"),
                "description": param_docs.get(name, ""),
            }
            if origin is list:
                # list[int] -> items {"type": "integer"}. A bare list has no
                # element type to read, so it falls back to string.
                element = next(iter(get_args(hint)), str)
                properties[name]["items"] = {
                    "type": PY_TO_JSON.get(element, "string")
                }
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "name": handler.__name__,
            "description": summary,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def register(self, name: str, handler: Callable):
        """Register a callable, validating its generated schema first.

        Raises pydantic.ValidationError if the handler's signature or
        docstring produce an invalid tool schema, so a broken tool fails
        at registration time instead of inside the agent loop.
        """
        schema = self._read_tool(handler)
        schema["name"] = name  # the registry name is authoritative for dispatch
        tool_schema = ToolSchema.model_validate(schema)
        self.tools[name] = Tool(name=name, tool_schema=tool_schema, handler=handler)

    def get_schemas(self) -> list[ToolSchema]:
        """Return neutral schemas for the provider to convert."""
        return [tool.tool_schema for tool in self.tools.values()]

    def dispatch(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string."""
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"
        try:
            result = tool.handler(**arguments)
            return str(result)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
