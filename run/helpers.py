import os
from time import time, sleep
from psutil import Process

from jax import clear_caches

def get_duration(t0, program='script'):
    t1 = time()
    duration = t1-t0; unit='s'
    if duration > 60.: duration=duration/60.; unit='min'
    if duration > 60.: duration=duration/60.; unit='h'
    print(f'\nDuration of {program}: {duration} {unit}\n\n')

def get_memory_usage():
        process = Process(os.getpid())
        mem = process.memory_info().rss  # in bytes
        print(f"Current memory usage: {mem / (1024 ** 2):.2f} MB")
        
def periodic_clear(interval):
    while True:
        sleep(interval)
        clear_caches()
        get_memory_usage()