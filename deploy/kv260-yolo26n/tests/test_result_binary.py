import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.render_result import parse_result_file


class ResultBinaryTests(unittest.TestCase):
    def test_reads_multiple_frames(self):
        payload = (struct.pack("<I6f", 1, 1, 2, 3, 4, 0.75, 2) +
                   struct.pack("<I", 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.bin"
            path.write_bytes(payload)
            self.assertEqual(parse_result_file(path), [[(1, 2, 3, 4, 0.75, 2)], []])

    def test_rejects_truncated_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.bin"
            path.write_bytes(struct.pack("<I", 1) + struct.pack("<2f", 1, 2))
            with self.assertRaisesRegex(ValueError, "Truncated detections"):
                parse_result_file(path)

    def test_rejects_more_than_maximum_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "too_many.bin"
            path.write_bytes(struct.pack("<I", 301))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                parse_result_file(path)


if __name__ == "__main__":
    unittest.main()
