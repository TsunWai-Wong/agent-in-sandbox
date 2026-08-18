import logging

from .base import Provider
from .openai_provider import OpenAIProvider


logger = logging.getLogger(__name__)


class ModelProviderRegistry:
    """Maps a provider name to its adapter class.

    Registration is explicit rather than discovered, so adding a provider is one
    line at the bottom of this module and no existing adapter is touched.
    """

    providers: dict[str, type[Provider]]

    def __init__(self) -> None:
        self.providers = {}

    def register(self, name: str, provider: type[Provider]) -> None:
        self.providers[name] = provider

    def get(self, name: str) -> type[Provider]:
        """Look up an adapter class, naming the alternatives when it is missing."""
        try:
            return self.providers[name]
        except KeyError:
            raise KeyError(
                f"Unknown model provider {name!r}. "
                f"Registered providers: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self.providers)


registry = ModelProviderRegistry()
registry.register("openai", OpenAIProvider)
