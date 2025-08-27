#!/usr/bin/env python3
import gi
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis

Aravis.update_device_list()
for i in range(Aravis.get_n_devices()):
    dev = Aravis.get_device_id(i)
    print(f"Device[{i}] = {dev}")