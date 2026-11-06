"""Web tools without the network: HTML to text, redirect unwrapping, private addresses refused."""

import pytest

from aroa.tools.base import ToolError
from aroa.tools.web import _unwrap, check_url, html_to_text


def test_html_to_text_drops_scripts_and_keeps_paragraphs():
    page = (
        "<html><head><title>x</title><style>p{}</style></head>"
        "<body><script>a()</script><h1>T</h1><p>One &amp; two</p><p>Three</p></body></html>"
    )
    assert html_to_text(page) == "T\nOne & two\nThree"


def test_duckduckgo_redirects_are_unwrapped():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.boe.es%2Fbuscar%2Fact.php%3Fid%3DBOE-A-1978-31229&rut=abc"
    assert _unwrap(wrapped) == "https://www.boe.es/buscar/act.php?id=BOE-A-1978-31229"
    assert _unwrap("https://example.org/page") == "https://example.org/page"


def test_private_addresses_are_refused():
    with pytest.raises(ToolError):
        check_url("http://127.0.0.1:8000/secret")
    with pytest.raises(ToolError):
        check_url("ftp://example.org/x")
    assert check_url("http://127.0.0.1:8000/x", allow_private=True) == "http://127.0.0.1:8000/x"
