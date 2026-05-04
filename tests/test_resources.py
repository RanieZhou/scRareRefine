import time
import unittest

from scrare_refine.resources import ResourceMonitor


class ResourceMonitorTests(unittest.TestCase):
    def test_monitor_records_wall_time_and_peak_memory(self):
        with ResourceMonitor(sample_interval_seconds=0.01) as monitor:
            payload = bytearray(1024 * 1024)
            payload[0] = 1
            time.sleep(0.03)

        summary = monitor.summary()

        self.assertGreater(summary["wall_time_seconds"], 0)
        self.assertGreater(summary["peak_rss_mb"], 0)


if __name__ == "__main__":
    unittest.main()
