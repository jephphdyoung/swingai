"""Mevo+ service discovery (UDP 1248, XML).

Broadcast a ServiceQuery; the unit replies with a ServiceInfo giving its Host and the
TCP control Port (5100). All plaintext.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from xml.etree import ElementTree

DISCOVERY_PORT = 1248
SERVICE_NAME = "FlightScope Device"
QUERY = (
    '<?xml version="1.0"?>'
    f"<ServiceQuery><ServiceName>{SERVICE_NAME}</ServiceName></ServiceQuery>"
).encode()


@dataclass(frozen=True)
class MevoDevice:
    host: str
    port: int
    instance: str | None = None  # e.g. "M2-047295"


def _parse_service_info(data: bytes) -> MevoDevice | None:
    try:
        root = ElementTree.fromstring(data.decode("utf-8", "replace"))
    except ElementTree.ParseError:
        return None
    if root.tag != "ServiceInfo":
        return None

    def _text(tag: str) -> str | None:
        el = root.find(tag)
        return el.text if el is not None else None

    host = _text("Host")
    port = _text("Port")
    if not host or not port:
        return None
    return MevoDevice(host=host, port=int(port), instance=_text("InstanceName"))


def discover(timeout: float = 3.0, broadcast_addr: str = "255.255.255.255") -> MevoDevice | None:
    """Broadcast a discovery query and return the first device that answers, or None.

    `broadcast_addr` can be set to a subnet broadcast (e.g. "192.168.120.255") if the
    global broadcast doesn't reach the device's VLAN.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    try:
        sock.sendto(QUERY, (broadcast_addr, DISCOVERY_PORT))
        try:
            while True:
                data, _addr = sock.recvfrom(4096)
                dev = _parse_service_info(data)
                if dev is not None:
                    return dev
        except socket.timeout:
            return None
    finally:
        sock.close()
