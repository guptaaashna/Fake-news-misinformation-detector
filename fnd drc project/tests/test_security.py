"""Mocked tests for URL security indicators."""

import socket
import ssl
import unittest
from unittest.mock import patch

from services.security import check_url


class SecurityCheckTests(unittest.TestCase):
	@patch("services.security._get_certificate", return_value={
		"issuer": "commonName=Example CA",
		"subject": "commonName=example.com",
		"expires": "Dec 31 23:59:59 2030 GMT",
	})
	@patch("services.security.socket.gethostbyname", return_value="93.184.216.34")
	def test_https_normal_domain(self, _, certificate):
		result = check_url("https://example.com/news")

		self.assertIsInstance(result, dict)
		self.assertTrue(result["https"])
		self.assertEqual(result["hostname"], "example.com")
		self.assertTrue(result["dns_resolves"])
		self.assertTrue(result["ssl_available"])
		self.assertFalse(result["suspicious_url"])
		self.assertEqual(result["threat_reputation"], "unknown")
		certificate.assert_called_once_with("example.com")

	@patch("services.security.socket.gethostbyname", return_value="93.184.216.34")
	def test_http_url_has_no_ssl_check(self, _):
		result = check_url("http://example.com/news")

		self.assertFalse(result["https"])
		self.assertTrue(result["dns_resolves"])
		self.assertFalse(result["ssl_available"])
		self.assertIn("URL does not use HTTPS.", result["indicators"])

	def test_invalid_url_is_safe(self):
		result = check_url("not-a-url")

		self.assertIsInstance(result, dict)
		self.assertEqual(result["hostname"], "unknown")
		self.assertTrue(result["suspicious_url"])

	def test_missing_hostname_is_safe(self):
		result = check_url("https:///missing-host")

		self.assertIsInstance(result, dict)
		self.assertEqual(result["hostname"], "unknown")
		self.assertFalse(result["dns_resolves"])

	@patch("services.security.socket.gethostbyname", side_effect=socket.gaierror("not found"))
	def test_dns_failure_does_not_crash(self, _):
		result = check_url("http://missing.example")

		self.assertFalse(result["dns_resolves"])
		self.assertIn("Hostname could not be resolved.", result["indicators"])

	def test_ip_address_is_suspicious_pattern(self):
		with patch("services.security.socket.gethostbyname", return_value="192.0.2.1"):
			result = check_url("http://192.0.2.1/article")

		self.assertTrue(result["suspicious_url"])
		self.assertIn("potentially suspicious pattern", " ".join(result["indicators"]))

	@patch("services.security.socket.gethostbyname", return_value="93.184.216.34")
	def test_long_hostname_is_suspicious_pattern(self, _):
		hostname = "one.two.three.four.five.example.com"
		result = check_url(f"http://{hostname}/article")

		self.assertTrue(result["suspicious_url"])

	@patch("services.security._get_certificate", side_effect=ssl.SSLError("certificate failure"))
	@patch("services.security.socket.gethostbyname", return_value="93.184.216.34")
	def test_ssl_failure_is_safe(self, _, certificate):
		result = check_url("https://example.com/news")

		self.assertFalse(result["ssl_available"])
		self.assertIn("SSL certificate information could not be retrieved.", result["indicators"])
		certificate.assert_called_once_with("example.com")

	@patch("services.security.socket.gethostbyname", return_value="93.184.216.34")
	def test_default_threat_reputation_is_unknown(self, _):
		result = check_url("http://example.com/news")

		self.assertEqual(result["threat_reputation"], "unknown")


if __name__ == "__main__":
	unittest.main()
