import threading
from dataclasses import dataclass
import time
from pynvml import *

from project_types.project_types import PhaseEnergy

@dataclass
class _Reading:
    timestamp: float
    power_w: float
    gpu_util_percent: float
    other_proc_count: int

class Energy_Monitor:
    def __init__(self, device_index: int = 0, poll_interval_s: float = 0.05):
        self.device_index = device_index
        self.poll_interval_s = poll_interval_s

        self._readings: list[_Reading] = []
        self._prefill_end_ts: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._own_pids: set[int] = set()


        nvmlInit()
        self._handle = nvmlDeviceGetHandleByIndex(device_index)


    def capture_baseline_pids(self) -> None:
        self._own_pids = {os.getpid()}
        try:
            for p in nvmlDeviceGetComputeRunningProcesses(self._handle):
                self._own_pids.add(p.pid)
        except NVMLError:
            pass


    def _other_process_count(self) -> int:
        try:
            procs = nvmlDeviceGetComputeRunningProcesses(self._handle)
        except NVMLError:
            return 0
        return sum(1 for p in procs if p.pid not in self._own_pids)

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

        energy = self._integrate(readings, split_ts)
        contention = self._summarize_contention(readings)
        return energy, contention    




    def _poll_loop(self) -> None:
        last_proc_check = 0.0
        other_count = 0
        while not self._stop_event.is_set():
            power_w = nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            util = nvmlDeviceGetUtilizationRates(self._handle)
            timestamp = time.perf_counter()

            if timestamp - last_proc_check > 0.5:
                other_count = self._other_process_count()
                last_proc_check = timestamp

            with self._lock:
                self._readings.append(_Reading(timestamp, power_w, util.gpu, other_count))
            time.sleep(self.poll_interval_s)



    @staticmethod
    def _trapezoid(readings: list[_Reading]) -> float:
        if len(readings) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(readings)):
            dt = readings[i].timestamp - readings[i - 1].timestamp
            avg_power = (readings[i].power_w + readings[i - 1].power_w) / 2.0
            total += dt * avg_power
        return total


    def _integrate(self, readings: list[_Reading], split_ts: float | None) -> PhaseEnergy:
        if not readings:
            return PhaseEnergy(0.0, 0.0, 0, 0)
        if split_ts is None:
            total = self._trapezoid(readings)
            return PhaseEnergy(0, total, 0)
        prefill_readings = [r for r in readings if r.timestamp < split_ts]
        decode_readings = [r for r in readings if r.timestamp >= split_ts]

        if prefill_readings and decode_readings:
            decode_readings =[prefill_readings[-1]] + decode_readings

        return PhaseEnergy(prefill_joules=self._trapezoid(prefill_readings), 
                           generation_joules=self._trapezoid(decode_readings), 
                           prefill_tokens=0, decode_tokens=0)

    @staticmethod
    def _summarize_contention(readings: list[_Reading]) -> dict:
        if not readings:
            return {"other_process_present_frac": 0.0, "max_other_process_count": 0, "mean_gpu_util_percent": 0.0}
        n_contended = sum(1 for r in readings if r.other_proc_count > 0)
        return {
            "other_process_present_frac": n_contended / len(readings),
            "max_other_process_count": max(r.other_proc_count for r in readings),
            "mean_gpu_util_percent": sum(r.gpu_util_percent for r in readings) / len(readings),
        }
