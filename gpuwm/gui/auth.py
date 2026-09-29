"""Who may talk to the GUI server.

- It binds to a loopback address unless ``--allow-remote`` is given.
- A 32-byte token is printed in the first link.  ``GET /?token=...`` puts
  it in an HttpOnly, SameSite=Strict cookie named per port and redirects
  to ``/`` without it, so the token leaves the address bar.
- Every POST carries the token in the ``X-ArWen-Token`` header.  The
  cookie alone never authorizes a POST, so a page on another site cannot
  make the browser press a button here (it cannot set that header).
- The Host header must name this server, and an Origin header, when
  present, must too: the guard against DNS rebinding.
- A peer that is not a loopback address is refused unless
  ``--allow-remote``.

A wrong Host or Origin is 400; a missing or wrong token, or a remote
peer, is 403.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hmac
import ipaddress
import secrets
import socket
from typing import Iterable, Mapping
from urllib.parse import urlsplit

TOKEN_BYTES = 32
TOKEN_HEADER = "X-ArWen-Token"
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]")
WILDCARD_BINDS = ("0.0.0.0", "::", "")


class BindRefused(ValueError):
    """The address the GUI was asked to listen on is not allowed."""

    def __init__(self, what: str, fix: str) -> None:
        super().__init__(f"{what} {fix}")
        self.what = what
        self.fix = fix


def new_token() -> str:
    """32 random bytes, URL-safe (43 characters)."""

    return secrets.token_urlsafe(TOKEN_BYTES)


def cookie_name(port: int) -> str:
    # Cookies are shared across ports on one host; a name per port keeps
    # two servers on one machine from overwriting each other's token.
    return f"arwen_token_{int(port)}"


def _ip(text: str):
    try:
        address = ipaddress.ip_address(text.strip("[]").split("%", 1)[0])
    except ValueError:
        return None
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def is_loopback(address: str) -> bool:
    ip = _ip(address)
    return bool(ip is not None and ip.is_loopback)


def resolve_bind(host: str) -> list[str]:
    """Every address ``host`` names, as the socket layer would bind it."""

    if host in WILDCARD_BINDS:
        return ["0.0.0.0" if host != "::" else "::"]
    if _ip(host) is not None:
        return [host.strip("[]")]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as error:
        raise BindRefused(f"The address {host!r} does not resolve ({error}).",
                          "Use --bind 127.0.0.1.") from error
    return sorted({str(info[4][0]) for info in infos})


def require_bind(host: str, allow_remote: bool) -> None:
    """Refuse a listening address other computers can reach, unless asked.

    The breakage it prevents: a server started with --bind 0.0.0.0 would
    let anyone on the network who learns the token start GPU jobs here.
    """

    if allow_remote:
        return
    remote = [address for address in resolve_bind(host) if not is_loopback(address)]
    if remote:
        raise BindRefused(
            f"--bind {host} listens on {', '.join(remote)}, which other computers can reach.",
            "Keep the default (127.0.0.1) and use ssh -L from the other computer, "
            "or pass --allow-remote if you mean it.")


def _host_port(value: str) -> str:
    return value.strip().lower().rstrip(".")


def allowed_hosts(port: int, bind: str = "127.0.0.1", allow_remote: bool = False,
                  extra: Iterable[str] = ()) -> frozenset[str]:
    """The Host header values this server answers to, lower case."""

    names = set(LOOPBACK_NAMES)
    if allow_remote:
        if bind not in WILDCARD_BINDS:
            names.add(f"[{bind}]" if ":" in bind and not bind.startswith("[") else bind)
        try:
            names.add(socket.gethostname())
            names.add(socket.getfqdn())
        except OSError:
            pass
    names.update(extra)
    hosts = {f"{_host_port(name)}:{int(port)}" for name in names if name}
    if int(port) == 80:
        hosts.update(_host_port(name) for name in names if name)
    return frozenset(hosts)


def parse_cookies(header: str | None) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in (header or "").split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name:
            cookies[name] = value.strip().strip('"')
    return cookies


@dataclass(frozen=True)
class Decision:
    """What the guard says about one request."""

    status: int = 200
    message: str = ""
    via: str = ""
    set_cookie: bool = False

    @property
    def allowed(self) -> bool:
        return self.status == 200


@dataclass(frozen=True)
class Guard:
    token: str
    port: int
    bind: str = "127.0.0.1"
    allow_remote: bool = False
    extra_hosts: tuple[str, ...] = ()
    hosts: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        if not self.hosts:
            object.__setattr__(self, "hosts", allowed_hosts(
                self.port, self.bind, self.allow_remote, self.extra_hosts))

    def host_ok(self, host: str | None) -> bool:
        return bool(host) and _host_port(host) in self.hosts

    def origin_ok(self, origin: str | None) -> bool:
        if origin is None:
            return True
        parts = urlsplit(origin.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return False
        return _host_port(parts.netloc) in self.hosts

    def peer_ok(self, peer: str) -> bool:
        return self.allow_remote or is_loopback(peer)

    def token_ok(self, candidate: str | None) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.encode("utf-8", "replace"), self.token.encode("utf-8"))

    def cookie_header(self) -> str:
        return f"{cookie_name(self.port)}={self.token}; Path=/; HttpOnly; SameSite=Strict"

    def check(self, method: str, path: str, headers: Mapping[str, str],
              query: Mapping[str, list[str]], peer: str) -> Decision:
        if not self.host_ok(headers.get("Host")):
            return Decision(400, "This server answers only to "
                                 f"localhost:{self.port} or 127.0.0.1:{self.port}.")
        if not self.origin_ok(headers.get("Origin")):
            return Decision(400, "The request came from another site's page.")
        if not self.peer_ok(peer):
            return Decision(403, "This server takes requests from this computer only. "
                                 "Use ssh -L from the other computer.")
        if self.token_ok(headers.get(TOKEN_HEADER)):
            return Decision(200, via="header")
        if method not in ("GET", "HEAD"):
            return Decision(403, f"A {method} needs the {TOKEN_HEADER} header.")
        # The first link is answered with the cookie and a redirect to a
        # bare "/", so the token never stays in the address bar or history.
        if path == "/" and self.token_ok((query.get("token") or [""])[0]):
            return Decision(200, via="query", set_cookie=True)
        if self.token_ok(parse_cookies(headers.get("Cookie")).get(cookie_name(self.port))):
            return Decision(200, via="cookie")
        return Decision(403, "Open the link gpuwm gui printed; it carries the token.")


__all__ = [
    "BindRefused", "Decision", "Guard", "TOKEN_BYTES", "TOKEN_HEADER",
    "allowed_hosts", "cookie_name", "is_loopback", "new_token",
    "parse_cookies", "require_bind", "resolve_bind",
]
