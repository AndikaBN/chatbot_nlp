import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp_dir.cleanup)
        fixture_path = Path(cls.temp_dir.name) / "dataset.csv"
        with fixture_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text", "label", "response"])
            writer.writerows(
                [
                    ["printer saya tidak bisa mencetak", "hardware", "Periksa koneksi printer."],
                    ["internet kantor lambat", "network", "Periksa sambungan jaringan."],
                    ["excel tidak bisa dibuka", "software", "Periksa aplikasi Excel."],
                ]
            )

        cls.classifier = Mock(return_value=[{"label": "hardware", "score": 0.95}])
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.pipeline = Mock(return_value=cls.classifier)
        original_open = Path.open
        configured_path = PROJECT_DIR / "data" / "dataset_chatbot_it_support_3_kategori.csv"

        def open_with_fixture(path, *args, **kwargs):
            if path == configured_path:
                path = fixture_path
            return original_open(path, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("api_under_test", PROJECT_DIR / "api.py")
        cls.api = importlib.util.module_from_spec(spec)
        original_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = transformers_stub
        try:
            with patch.object(Path, "open", open_with_fixture):
                spec.loader.exec_module(cls.api)
        finally:
            if original_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = original_transformers
        cls.client = TestClient(cls.api.app)
        cls.addClassCleanup(cls.client.close)

    def setUp(self):
        self.classifier.reset_mock()

    def test_greetings_and_short_messages_bypass_classifier(self):
        cases = {
            "hai": "Halo!",
            "HALLO!!!": "Halo!",
            "halooo min": "Halo!",
            "hello bot": "Halo!",
            "salam": "Halo!",
            "salam sejahtera": "Halo!",
            "Assalamualaikum": "Waalaikumsalam!",
            "Assalamu'alaikum wr. wb.": "Waalaikumsalam!",
            "assalamualaikum warahmatullahi wabarakatuh": "Waalaikumsalam!",
            "Selamat pagi, kak!": "Selamat pagi!",
            "siang min": "Selamat siang!",
            "selamat sore": "Selamat sore!",
            "selamat malam": "Selamat malam!",
            "test": "Pesan tes diterima.",
            "TES 123": "Pesan tes diterima.",
            "ping": "Pesan tes diterima.",
            "terima kasih": "Sama-sama!",
            "makasih min": "Sama-sama!",
            "help": "Saya dapat membantu",
            "bisa bantu apa?": "Saya dapat membantu",
            "oke": "Baik.",
            "sampai jumpa": "Sampai jumpa!",
            "apa kabar?": "Saya siap membantu",
        }
        for message, expected_reply in cases.items():
            with self.subTest(message=message):
                response = self.client.post("/chat", json={"message": message, "debug": True})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertTrue(result["success"])
                self.assertIn(expected_reply, result["reply"])
                self.assertIsNone(result["intent"])
                self.assertIsNone(result["confidence"])
                self.assertIsNone(result["matched_question"])
                self.assertEqual(result["top_matches"], [])
                self.assertFalse(result["should_ask_resolution"])
        self.classifier.assert_not_called()

    def test_short_message_with_complaint_uses_it_support(self):
        for message in [
            "halo, printer saya tidak bisa mencetak",
            "selamat pagi, printer saya tidak bisa mencetak",
            "test printer saya tidak bisa mencetak",
            "terima kasih, printer saya tidak bisa mencetak",
        ]:
            with self.subTest(message=message):
                self.classifier.reset_mock()
                response = self.client.post("/chat", json={"message": message})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["reply"], "Periksa koneksi printer.")
                self.assertEqual(result["retrieval_label"], "hardware")
                self.assertTrue(result["should_ask_resolution"])
                self.classifier.assert_called_once_with(message)

    def test_unknown_message_keeps_existing_fallback(self):
        response = self.client.post("/chat", json={"message": "abrakadabra"})
        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("belum menemukan jawaban", result["reply"])
        self.assertFalse(result["should_ask_resolution"])
        self.classifier.assert_called_once_with("abrakadabra")

    def test_whitespace_or_punctuation_asks_for_message(self):
        for message in ["   ", "!!!"]:
            with self.subTest(message=message):
                response = self.client.post("/chat", json={"message": message})
                self.assertEqual(response.status_code, 200)
                self.assertIn("Pesan masih kosong", response.json()["reply"])
                self.assertFalse(response.json()["should_ask_resolution"])
        self.classifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
