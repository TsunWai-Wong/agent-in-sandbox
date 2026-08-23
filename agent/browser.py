import functools

from playwright.sync_api import sync_playwright


HEADLESS = False           # watch it work while you debug

# How long an action waits for its element before giving up. This bounds page
# and element waits only -- it has nothing to do with how long the model takes
# to decide on a call, so raising it will not help a slow model.
ACTION_TIMEOUT_MS = 15000

# Tag every visible interactive element with a number, then report
# role + accessible name. Claude refers to elements by that number.

INDEX_JS = """
() => {
  document.querySelectorAll('[data-ref]').forEach(e => e.removeAttribute('data-ref'));

  const SEL = [
    'a[href]', 'button', 'input', 'select', 'textarea',
    '[role=button]', '[role=link]', '[role=textbox]', '[role=checkbox]',
    '[role=radio]', '[role=tab]', '[role=menuitem]', '[role=combobox]',
    '[contenteditable=true]', '[onclick]'
  ].join(',');

  const out = [];
  let i = 0;

  for (const el of document.querySelectorAll(SEL)) {
    const box = el.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) continue;

    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') continue;

    const name = (
      el.getAttribute('aria-label') ||
      el.innerText ||
      el.value ||
      el.placeholder ||
      el.getAttribute('title') ||
      el.alt || ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 90);

    let role = el.getAttribute('role') || el.tagName.toLowerCase();
    if (el.tagName === 'INPUT' && el.type) role += ':' + el.type;

    el.setAttribute('data-ref', String(i));
    out.push({
      ref: i,
      role: role,
      name: name,
      disabled: !!el.disabled,
      offscreen: box.top > window.innerHeight || box.bottom < 0
    });
    i++;
  }
  return out;
}
"""


def snapshot_on_error(fn):
    """Pair any failure from an action with a fresh snapshot.

    ToolRegistry.dispatch reduces an exception to a bare error string, which
    leaves the model holding refs from a page that has since moved on. A failed
    click is recoverable only if the error arrives with the page it happened on.

    functools.wraps matters here beyond tidiness: ToolRegistry builds a tool's
    schema from the handler's signature, docstring and annotations, all of which
    it copies (and inspect.signature follows __wrapped__ past the *args).
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # fn is the undecorated function, so self is passed through explicitly.
        return self._guard(fn, self, *args, **kwargs)

    return wrapper


class Browser:
    def __init__(self, headless=HEADLESS, viewport=None):
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 900}
        self._pw = None
        self._browser = None
        self._page = None
        # The previous snapshot, so a no-op action can be reported as one.
        self._last_snapshot = None

    @property
    def page(self):
        """The Playwright page, launching the browser on first access."""
        if self._page is None:
            self.start()
        return self._page

    def start(self):
        """Launch Playwright and open a page. Idempotent."""
        if self._page is not None:
            return self
        try:
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page(viewport=self.viewport)
        except Exception:
            self.close()   # idempotent; drops a half-built driver
            raise
        return self

    def close(self):
        """Close the browser and stop Playwright. Idempotent."""
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
            self._pw = self._browser = self._page = None
            # Or a restarted browser would compare its first page against the
            # last page of the previous session.
            self._last_snapshot = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    def _node(self, ref):
        return self.page.locator(f'[data-ref="{ref}"]').first

    def _guard(self, fn, *args, **kwargs):
        """Run an action, returning the error plus a snapshot instead of raising."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            try:
                return f"{type(e).__name__}: {e}\n\n{self.snapshot()}"
            except Exception:
                # The page itself is gone — a crash, or a closed browser. Report
                # the original failure rather than the one from re-reading it.
                return f"{type(e).__name__}: {e}\n\n(snapshot unavailable)"

    def snapshot(self) -> str:
        """Build a text snapshot of the current page for Claude.

        Returns:
            str: URL, title, interactive elements, and page text.
        """
        self.page.wait_for_load_state("domcontentloaded")
        elements = self.page.evaluate(INDEX_JS)

        lines = []
        for e in elements:
            flags = ""
            if e["disabled"]:
                flags += " [disabled]"
            if e["offscreen"]:
                flags += " [offscreen]"
            lines.append(f'[{e["ref"]}] {e["role"]} "{e["name"]}"{flags}')

        text = self.page.evaluate("document.body.innerText || ''").strip()
        text = "\n".join(l for l in text.splitlines() if l.strip())[:3000]

        snap = (
            f"URL: {self.page.url}\n"
            f"Title: {self.page.title()}\n\n"
            f"Interactive elements:\n" + ("\n".join(lines) or "(none)") + "\n\n"
            f"Page text:\n{text}"
        )

        # A click that lands on the page it started from returns bytes identical
        # to the previous snapshot, which reads to the model as "it hasn't
        # happened yet" and invites the same call again. Say so instead.
        if snap == self._last_snapshot:
            decorated = (
                "(No change from the previous snapshot — that action had no "
                "visible effect. Try something else.)\n\n" + snap
            )
        else:
            decorated = snap

        # The bare snapshot is what gets stored, never the decorated one, or the
        # third identical call would compare against a prefixed string, miss,
        # and drop the warning at exactly the point the loop is established.
        self._last_snapshot = snap
        return decorated

    @snapshot_on_error
    def navigate(self, url: str):
        """Navigate to a URL and return the resulting page snapshot.

        Args:
            url (str): URL to navigate to.

        Returns:
            str: Snapshot of the current page state.
        """
        self.page.goto(url, wait_until="domcontentloaded")
        return self.snapshot()


    @snapshot_on_error
    def click(self, ref: int):
        """Click a referenced page element and return a snapshot.

        Args:
            ref (int): Reference identifying the element to click.

        Returns:
            str: Snapshot of the updated page state.
        """
        self._node(ref).click(timeout=ACTION_TIMEOUT_MS)
        self.page.wait_for_timeout(500)
        return self.snapshot()


    @snapshot_on_error
    def fill(self, ref: int, text: str, submit: bool = False):
        """Fill a referenced input and optionally submit it.

        Args:
            ref (int): Reference identifying the input element.
            text (str): Text to enter into the input.
            submit (bool): Whether to press Enter after filling.

        Returns:
            str: Snapshot of the updated page state.
        """
        node = self._node(ref)
        node.fill(text, timeout=ACTION_TIMEOUT_MS)
        if submit:
            node.press("Enter")
        self.page.wait_for_timeout(500)
        return self.snapshot()


    @snapshot_on_error
    def select(self, ref: int, value: str):
        """Select an option from a referenced element.

        Args:
            ref (int): Reference identifying the select element.
            value (str): Option value to select.

        Returns:
            str: Snapshot of the updated page state.
        """
        self._node(ref).select_option(value, timeout=ACTION_TIMEOUT_MS)
        return self.snapshot()


    @snapshot_on_error
    def press(self, key: str):
        """Press a keyboard key and return the page snapshot.

        Args:
            key (str): Key to press.

        Returns:
            str: Snapshot of the updated page state.
        """
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(300)
        return self.snapshot()


    @snapshot_on_error
    def scroll(self, direction: str = "down"):
        """Scroll the page in the specified direction.

        Args:
            direction (str): Scroll direction, either "down" or "up".

        Returns:
            str: Snapshot of the updated page state.
        """
        delta = 600 if direction == "down" else -600
        self.page.mouse.wheel(0, delta)
        self.page.wait_for_timeout(300)
        return self.snapshot()


    @snapshot_on_error
    def back(self):
        """Navigate back to the previous page and return its snapshot.

        Returns:
            str: Snapshot of the previous page state.
        """
        self.page.go_back(wait_until="domcontentloaded")
        return self.snapshot()
