# Sourced from https://gist.github.com/thomasbrandon/a1e126de770c7e04f8d71a7dc971cfb7

import numpy as np
from math import ceil
import torch as torch
from time import perf_counter


class CPUEvent:
    def __init__(self, enable_timing=True):
        """
        Constructor
        :param enable_timing:
        """
        self.time = None

    def record(self, stream=None):
        """
        Record an event
        :param stream:
        :return:
        """
        self.time = perf_counter()

    def elapsed_time(self, end_event):
        """
        Time elapsed
        :param end_event:
        :return:
        """
        if self.time == None:
            raise RuntimeError("This event was not recorded.")
        if end_event.time == None:
            raise RuntimeError("The since event was not recorded.")
        return (end_event.time - self.time) * 1000

    def wait(self, stream=None):
        """
        Wait
        :param stream:
        :return:
        """
        pass  # Dummy wait for CPU


class CPUStream:
    def record_event(self, event=None):
        if event is None:
            event = CPUEvent()
        event.record()
        return event


class Profiler:
    """
    This class lets you profile PyTorch code on both CPU and GPU. Use it as a context manager.
    You call `start` after initialisation code to exclude that from timing while still having it
    run on the appropriate device.
    You can use `record` to record intermediate times in a process. In this case `get_results` will return
    a list of times which are the times taken between successive recorded times, not cumulative times.
    Times are returned in milliseconds (with fractions) as with the GPU timer. GPU timing resolution is about
    1/2 a microsecond according to nvidia. CPU timing resolution is determined by pythons `time.perf_counter`
    function which depends on available system clocks but should give nanosecond resolution.
    """
    def __init__(self, device='cpu', wait=False):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.wait = wait
        self.events = []
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=self.device)
            self.event_cls = torch.cuda.Event
        else:
            self.stream = CPUStream()
            self.event_cls = CPUEvent
        # self.cur_event_id = None
        # self._event_cls = CPUEvent if device.type == 'cpu' else partial(torch.cuda.Event, enable_timing=True)

    def __enter__(self):
        if self.device.type == 'cuda':
            self.stream_cm = torch.cuda.stream(self.stream)
            self.stream_cm.__enter__()
        else:
            self.stream_cm = None
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, tb):
        # Record final event
        self.record()
        if self.stream_cm:
            self.stream.synchronize()
            # self.times = [self.events[i].elapsed_time(self.events[i+1]) for i in range(len(self.events) - 1)]
            self.stream_cm.__exit__(exc_type, exc_val, tb)
            self.stream_cm = None

    def start(self):
        self.events.clear()
        self.record()

    def record(self, event=None):
        if event is None:
            event = self.event_cls(enable_timing=True)
        self.stream.record_event(event)
        self.events.append(event)
        if self.wait: event.wait(self.stream)

    def get_results(self, total=False, reset=False):
        # if self.times: return self.times
        times = [self.events[i].elapsed_time(self.events[i + 1]) for i in range(len(self.events) - 1)]
        if total and len(times) > 1:
            times.insert(0, self.events[0].elapsed_time(self.events[-1]))
        if reset: self.events = []
        return times


def profile(func, *funcargs, device='cpu', wait=False, n_repeats=100, total=False, warmup=10, **func_kwargs):
    """
    Profile `func` on `device` across `n_repeats` runs after `warmup` runs.
    As well as `func_args` and `func_kwargs`additional `prof` and `device` keyword arguments wll be passed.
    You can call `prof.start` after any initial processing that should not be included in timing. Call `prof.record`
    to record timing for intermediate events. If you pass `wait==True` and are running on GPU then at each call to `prof.record`
    all previous work will be completed before moving on, otherwise processing from previous PyTorch calls may still be in progress.
    Returns an numpy array of timings for each run. If `prof.record` is called then multiple sets of timing data are recorded indicating the
    time between each recorded event. If `total==True` then also returns total time.
    :param func:
    :param funcargs:
    :param device:
    :param wait:
    :param n_repeats:
    :param total:
    :param warmup:
    :param func_kwargs:
    :return:
    """

    prof = Profiler(device=device, wait=wait)
    times = []
    for i in range(n_repeats + warmup):
        with prof:
            func(*funcargs, prof=prof, device=device, **func_kwargs)
        if i >= warmup: times.append(prof.get_results(total=total, reset=True))
    res = np.stack([np.array(t) for t in times], axis=1)
    return res


class STFTUtils:
    @staticmethod
    def __get_samples(num, length, dtype=np.float32):
        return [np.random.random((ceil(length))).astype(dtype) for _ in range(num)]

    @staticmethod
    def torch_stft(bs=64, sample_len=50 * 1024, n_fft=2048, win_length=512, copyto=False, copyfrom=False,
                   device='cpu', prof=None):
        if isinstance(device, str): device = torch.device(device)
        samples = STFTUtils.__get_samples(bs, sample_len)
        window = torch.hann_window(n_fft, device=device)
        batch = torch.tensor(np.stack(samples), device=(device if not copyto else 'cpu'))
        if prof: prof.start()
        if copyto and device.type != 'cpu':
            batch = batch.to(device)
        res = torch.stft(batch, n_fft, hop_length=win_length, window=window)
        if copyfrom and device.type != 'cpu':
            res = res[...,-1].to('cpu')
