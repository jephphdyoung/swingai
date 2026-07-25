"""Offline source: replay the device->host TCP 5100 stream from a pcap.

Lets the frame/metric decode be tested without the live device. Uses scapy's streaming
PcapReader (the full-swing captures are 100+ MB, so we never load the whole file).

The captures put the Mevo at 192.168.120.169; override with `mevo_ip` if needed.
"""

from __future__ import annotations

from collections.abc import Iterator

from .frames import Deframer, Frame

DEFAULT_MEVO_IP = "192.168.120.169"
CONTROL_PORT = 5100


def iter_device_payloads(pcap_path: str, mevo_ip: str = DEFAULT_MEVO_IP) -> Iterator[bytes]:
    """Yield TCP payloads sent *by the device* from port 5100, in capture order."""
    from scapy.all import IP, TCP, PcapReader  # imported lazily; scapy is heavy

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP not in pkt or TCP not in pkt:
                continue
            ip, tcp = pkt[IP], pkt[TCP]
            if ip.src != mevo_ip or tcp.sport != CONTROL_PORT:
                continue
            data = bytes(tcp.payload)
            if data:
                yield data


def iter_frames(pcap_path: str, mevo_ip: str = DEFAULT_MEVO_IP) -> Iterator[Frame]:
    """Deframe the device->host 5100 stream from a pcap into Frames."""
    deframer = Deframer()
    for payload in iter_device_payloads(pcap_path, mevo_ip):
        yield from deframer.feed(payload)
