import threading
from dataclasses import dataclass
import time
from pynvml import *

from project_types.project_types import PhaseEnergy

@dataclass
class _Reading:
    timestamp: float
    power_w: float

class Energy_Monitor:
    def __init__(self, device_index: int = 0, poll_interval_s: float = 0.05):
        self.device_index = device_index
        self.poll_interval_s = poll_interval_s

        self._readings: list[_Reading] = []
        self._prefill_end_ts: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        nvmlInit()
        self._handle = nvmlDeviceGetHandleByIndex(device_index)


    def start(self):
        self._readings.clear()
        self._prefill_end_ts = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()


    def mark_prefill_end(self):
        #this is for phase differentiation of pre and decode phases
        with self._lock:
            self._prefill_end_ts = time.perf_counter()


    def stop(self) -> PhaseEnergy:
        self._stop_event.set()
        if self._thread:
            self._thread.join()

        with self._lock:
            readings = list(self._readings)
            split_ts = self._prefill_end_ts

        return self._integrate(readings, split_ts)    




    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            power_mw = nvmlDeviceGetPowerUsage(self._handle)
            power_w = power_mw / 1000.0
            timestamp = time.perf_counter()
            with self._lock:
                self._readings.append(_Reading(timestamp, power_w))
            time.sleep(self.poll_interval_s)


    #TODO: implement method to measure joules




    #TODO: implement method to measure FLOPs




    #TODO: implement method to propagate PhaseEnergy