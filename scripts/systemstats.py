#!/usr/bin/env python3
import psutil
import GPUtil
import time
import curses
import os

def get_network_usage(interval=1.0):
    """Return network usage in KB/s for each interface."""
    before = psutil.net_io_counters(pernic=True)
    time.sleep(interval)
    after = psutil.net_io_counters(pernic=True)

    usage = {}
    for iface in before.keys():
        sent = (after[iface].bytes_sent - before[iface].bytes_sent) / 1024.0 / interval
        recv = (after[iface].bytes_recv - before[iface].bytes_recv) / 1024.0 / interval
        usage[iface] = (recv, sent)
    return usage

def draw_dashboard(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)

    while True:
        stdscr.erase()

        # CPU usage
        cpu_percents = psutil.cpu_percent(percpu=True)
        stdscr.addstr(0, 0, "CPU Usage per Core:")
        for i, p in enumerate(cpu_percents):
            stdscr.addstr(1 + i, 2, f"Core {i}: {p:5.1f}%")

        # Memory usage
        mem = psutil.virtual_memory()
        line = len(cpu_percents) + 2
        stdscr.addstr(line, 0, "Memory Usage:")
        stdscr.addstr(line + 1, 2,
            f"Used: {mem.used/1024/1024:.1f} MB / {mem.total/1024/1024:.1f} MB "
            f"({mem.percent}%)")
        stdscr.addstr(line + 2, 2,
            f"Available: {mem.available/1024/1024:.1f} MB | "
            f"Buffers: {getattr(mem, 'buffers', 0)/1024/1024:.1f} MB | "
            f"Cache: {getattr(mem, 'cached', 0)/1024/1024:.1f} MB")

        # Network usage
        net_usage = get_network_usage(0.5)
        line += 4
        stdscr.addstr(line, 0, "Network Usage (KB/s):")
        for j, (iface, (rx, tx)) in enumerate(net_usage.items()):
            stdscr.addstr(line + 1 + j, 2, f"{iface:10s} RX: {rx:8.1f} KB/s | TX: {tx:8.1f} KB/s")

        # GPU usage
        gpus = GPUtil.getGPUs()
        line += len(net_usage) + 2
        if gpus:
            stdscr.addstr(line, 0, "NVIDIA GPUs:")
            for k, gpu in enumerate(gpus):
                stdscr.addstr(line + 1 + k, 2,
                    f"{gpu.name} | Load: {gpu.load*100:5.1f}% | "
                    f"Mem: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB | "
                    f"Temp: {gpu.temperature}°C")
        else:
            stdscr.addstr(line, 0, "No NVIDIA GPUs found")

        stdscr.refresh()

        # Press 'q' to quit
        try:
            if stdscr.getkey() == "q":
                break
        except:
            pass

def main():
    curses.wrapper(draw_dashboard)

if __name__ == "__main__":
    main()
