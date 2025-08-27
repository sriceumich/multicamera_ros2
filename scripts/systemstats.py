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
        max_y, max_x = stdscr.getmaxyx()

        def safe_addstr(y, x, text):
            if y < max_y and x < max_x:
                stdscr.addstr(y, x, text[: max_x - x])

        # CPU usage (horizontal layout)
        cpu_percents = psutil.cpu_percent(percpu=True)
        safe_addstr(0, 0, "CPU Usage per Core:")

        col_width = 15  # width reserved per core entry
        cols = max_x // col_width
        for i, p in enumerate(cpu_percents):
            row = 1 + (i // cols)
            col = (i % cols) * col_width
            safe_addstr(row, col, f"Core {i:2d}: {p:5.1f}%")

        # Memory usage
        line = 2 + (len(cpu_percents) + cols - 1) // cols
        mem = psutil.virtual_memory()
        safe_addstr(line, 0, "Memory Usage:")
        safe_addstr(line + 1, 2,
            f"Used: {mem.used/1024/1024:.1f} MB / {mem.total/1024/1024:.1f} MB ({mem.percent}%)")
        safe_addstr(line + 2, 2,
            f"Available: {mem.available/1024/1024:.1f} MB | "
            f"Buffers: {getattr(mem, 'buffers', 0)/1024/1024:.1f} MB | "
            f"Cache: {getattr(mem, 'cached', 0)/1024/1024:.1f} MB")

        # Network usage
        net_usage = get_network_usage(0.5)
        line += 4
        safe_addstr(line, 0, "Network Usage (KB/s):")
        for j, (iface, (rx, tx)) in enumerate(net_usage.items()):
            safe_addstr(line + 1 + j, 2,
                f"{iface:10s} RX: {rx:8.1f} KB/s | TX: {tx:8.1f} KB/s")

        # GPU usage
        gpus = GPUtil.getGPUs()
        line += len(net_usage) + 2
        if gpus:
            safe_addstr(line, 0, "NVIDIA GPUs:")
            for k, gpu in enumerate(gpus):
                safe_addstr(line + 1 + k, 2,
                    f"{gpu.name} | Load: {gpu.load*100:5.1f}% | "
                    f"Mem: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB | "
                    f"Temp: {gpu.temperature}°C")
        else:
            safe_addstr(line, 0, "No NVIDIA GPUs found")

        stdscr.refresh()

        try:
            if stdscr.getkey() == "q":
                break
        except:
            pass


def main():
    curses.wrapper(draw_dashboard)

if __name__ == "__main__":
    main()
