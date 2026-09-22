"""Basic URL and connection security indicators."""

import ipaddress
import socket
import ssl
from urllib.parse import urlparse


SECURITY_TIMEOUT = 5


def _unknown_result() -> dict:
	return {
		"https": False,
		"suspicious_url": False,
		"hostname": "unknown",
		"dns_resolves": False,
		"ssl_available": False,
		"certificate": {"issuer": "unknown", "subject": "unknown", "expires": "unknown"},
		"threat_reputation": "unknown",
		"indicators": [],
	}


def _certificate_value(value: tuple) -> str:
	parts = []
	for group in value or ():
		for key, item in group:
			parts.append(f"{key}={item}")
	return ", ".join(parts) or "unknown"


def _get_certificate(hostname: str) -> dict[str, str]:
	"""Retrieve a peer certificate using normal certificate verification."""
	context = ssl.create_default_context()
	with socket.create_connection((hostname, 443), timeout=SECURITY_TIMEOUT) as connection:
		with context.wrap_socket(connection, server_hostname=hostname) as secure_socket:
			certificate = secure_socket.getpeercert()
	return {
		"issuer": _certificate_value(certificate.get("issuer")),
		"subject": _certificate_value(certificate.get("subject")),
		"expires": certificate.get("notAfter", "unknown"),
	}


def _has_suspicious_pattern(parsed_url, hostname: str) -> bool:
	if not hostname:
		return True
	try:
		ipaddress.ip_address(hostname)
		return True
	except ValueError:
		pass
	return any(
		[
			len(hostname) > 100,
			hostname.count(".") >= 4,
			"@" in parsed_url.netloc,
			"%" in parsed_url.netloc or "%" in parsed_url.path,
			any(character in hostname for character in ("_", " ", "\\")),
		]
	)


def check_url(url: str) -> dict:
	"""Collect raw security indicators for one HTTP(S) URL."""
	result = _unknown_result()
	parsed_url = urlparse((url or "").strip())
	result["https"] = parsed_url.scheme == "https"
	hostname = parsed_url.hostname
	if parsed_url.scheme not in {"http", "https"} or not hostname:
		result["suspicious_url"] = True
		result["indicators"].append("URL does not have a valid HTTP or HTTPS hostname.")
		return result

	result["hostname"] = hostname
	result["suspicious_url"] = _has_suspicious_pattern(parsed_url, hostname)
	if result["https"]:
		result["indicators"].append("HTTPS is enabled.")
	else:
		result["indicators"].append("URL does not use HTTPS.")

	try:
		socket.gethostbyname(hostname)
		result["dns_resolves"] = True
		result["indicators"].append("Hostname resolves successfully.")
	except (socket.gaierror, OSError):
		result["indicators"].append("Hostname could not be resolved.")

	if result["https"]:
		try:
			result["certificate"] = _get_certificate(hostname)
			result["ssl_available"] = True
			result["indicators"].append("SSL certificate information was retrieved.")
		except (OSError, ssl.SSLError, socket.timeout):
			result["indicators"].append("SSL certificate information could not be retrieved.")
	else:
		result["indicators"].append(
			"SSL certificate information is unavailable because the URL does not use HTTPS."
		)

	if result["suspicious_url"]:
		result["indicators"].append("URL contains a potentially suspicious pattern.")
	else:
		result["indicators"].append("No obvious suspicious URL pattern was detected.")
	result["indicators"].append("Threat reputation is unavailable.")
	return result
