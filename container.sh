#!/bin/bash
export UID=$(id -u)
export GID=$(id -g)
python3 container_ui.py
