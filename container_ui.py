import tkinter as tk
from tkinter import messagebox, scrolledtext, Toplevel, filedialog, simpledialog, StringVar
import docker
import yaml
import os
import subprocess
from threading import Thread
import time

client = docker.from_env()

compose_file = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
with open(compose_file, 'r') as f:
    compose = yaml.safe_load(f)

# Parse base compose services
services = compose.get('services', {})
container_map = {v.get('container_name', k): v for k, v in services.items()}
service_map = {v.get('container_name', k): k for k, v in services.items()}

# Detect the most recent 'docker-goal' container
latest_goal = None
for c in sorted(client.containers.list(all=True), key=lambda x: x.attrs['Created'], reverse=True):
    if c.name.startswith("docker-goal"):
        latest_goal = c
        break

# Replace 'goal' mapping with the dynamic container
if latest_goal:
    goal_service_key = "goal"
    container_map[goal_service_key] = {
        "dynamic": True,
        "image": latest_goal.image.tags[0] if latest_goal.image.tags else "goal_publisher:latest",
        "name": latest_goal.name
    }
    service_map[goal_service_key] = goal_service_key
    container_names = list(container_map.keys())
    has_goal_publisher = True
else:
    container_names = list(container_map.keys())
    has_goal_publisher = "goal" in container_names

scalable_keywords = ['agent', 'worker', 'node'] 

status_labels = {}

def resolve_container_name(name):
    if name == "goal":
        resolved = resolve_goal_container()
        if resolved:
            return resolved
    return name

def resolve_goal_container():
    containers = [
        c for c in client.containers.list(all=True)
        if c.name.startswith("docker-goal")
    ]
    if containers:
        # Prefer the newest running container
        containers.sort(key=lambda c: c.attrs["Created"], reverse=True)
        return containers[0].name
    return None


def copy_files(name):
    container_name = resolve_container_name(name)
    if not container_name:
        messagebox.showerror("Error", f"No container found for {name}")
        return
    def browse_host_src():
        path = filedialog.askopenfilename(title="Select file") or filedialog.askdirectory(title="Select folder")
        if path:
            host_src_entry.delete(0, tk.END)
            host_src_entry.insert(0, path)

    def browse_host_dest():
        path = filedialog.askdirectory(title="Select destination folder")
        if path:
            host_dest_entry.delete(0, tk.END)
            host_dest_entry.insert(0, path)

    def browse_container_path(entry_field):
        browser = Toplevel(root)
        browser.title("Browse Container Files")

        current_path = StringVar(value="/")

        def list_contents(path):
            try:
                result = subprocess.check_output(
                    ["docker", "exec", container_name, "bash", "-c", f"ls -p {path}"],
                    stderr=subprocess.STDOUT,
                    text=True
                )
                entries = result.strip().split("\n")
                entries = sorted(entries, key=lambda e: (not e.endswith("/"), e.lower()))
                return entries
            except subprocess.CalledProcessError as e:
                return [f"Error: {e.output.strip()}"]

        def refresh(path):
            listbox.delete(0, tk.END)
            contents = list_contents(path)
            for entry in contents:
                listbox.insert(tk.END, entry)

        def on_select(event):
            selection = listbox.curselection()
            if not selection:
                return
            selected = listbox.get(selection[0])
            path = current_path.get()
            if selected.endswith("/"):
                new_path = os.path.join(path, selected.strip("/"))
                if not new_path.endswith("/"):
                    new_path += "/"
                current_path.set(new_path)
                refresh(new_path)
            else:
                full_path = os.path.join(current_path.get(), selected)
                entry_field.delete(0, tk.END)
                entry_field.insert(0, full_path)
                browser.destroy()

        def go_up():
            current = current_path.get().rstrip("/")
            parent = os.path.dirname(current)
            if not parent:
                parent = "/"
            elif not parent.endswith("/"):
                parent += "/"
            current_path.set(parent)
            refresh(parent)

        top = tk.Frame(browser)
        top.pack(fill=tk.X)
        tk.Entry(top, textvariable=current_path, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(top, text="↑", command=go_up).pack(side=tk.RIGHT)

        listbox = tk.Listbox(browser, width=80, height=25)
        listbox.pack(fill=tk.BOTH, expand=True)
        listbox.bind('<<ListboxSelect>>', on_select)

        refresh(current_path.get())

    def browse_container_path2(entry_field):
        win_ls = Toplevel(root)
        win_ls.title("Browse Container")

        path_var = tk.StringVar(value="/")
        result_box = scrolledtext.ScrolledText(win_ls, width=80, height=20)
        result_box.pack()

        def refresh_listing():
            path = path_var.get()
            try:
                output = subprocess.check_output(
                    ["docker", "exec", name, "ls", "-l", path],
                    stderr=subprocess.STDOUT,
                    text=True
                )
                result_box.delete("1.0", tk.END)
                result_box.insert(tk.END, f"Browsing: {path}\n\n")
                entries = output.splitlines()
                for line in entries:
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    perms, _, _, _, _, _, _, _, fname = parts[:9]
                    full_path = os.path.join(path, fname)
                    if perms.startswith("d"):
                        result_box.insert(tk.END, f"[DIR] {fname}\n")
                    else:
                        result_box.insert(tk.END, f"      {fname}\n")
            except subprocess.CalledProcessError as e:
                result_box.delete("1.0", tk.END)
                result_box.insert(tk.END, f"Error: {e.output}")

        def go_to_path():
            refresh_listing()

        def set_selected():
            entry_field.delete(0, tk.END)
            entry_field.insert(0, path_var.get())
            win_ls.destroy()

        path_frame = tk.Frame(win_ls)
        path_frame.pack(pady=5)

        tk.Label(path_frame, text="Path:").pack(side=tk.LEFT)
        path_entry = tk.Entry(path_frame, textvariable=path_var, width=60)
        path_entry.pack(side=tk.LEFT)
        tk.Button(path_frame, text="Go", command=go_to_path).pack(side=tk.LEFT)

        nav_frame = tk.Frame(win_ls)
        nav_frame.pack()

        def go_up():
            current = path_var.get()
            if current != "/":
                new_path = os.path.dirname(current.rstrip("/"))
                path_var.set(new_path or "/")
                refresh_listing()

        tk.Button(nav_frame, text="↑ Up", command=go_up).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="✓ Select This Path", command=set_selected).pack(side=tk.LEFT, padx=5)

        refresh_listing()


    def host_to_container():
        src = host_src_entry.get()
        dest = container_dest_entry.get()
        if not os.path.exists(src):
            messagebox.showerror("Error", "Source path does not exist.")
            return
        try:
            subprocess.run(["docker", "cp", src, f"{container_name}:{dest}"], check=True)
            # messagebox.showinfo("Copy", "Copied from host to container.")
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed:\n{e}")

    def container_to_host():
        src = container_src_entry.get()
        dest = host_dest_entry.get()
        try:
            subprocess.run(["docker", "cp", f"{container_name}:{src}", dest], check=True)
            # messagebox.showinfo("Copy", "Copied from container to host.")
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed:\n{e}")

    win = Toplevel(root)
    win.title(f"Copy: {container_name}")

    # Host → Container
    tk.Label(win, text="Host → Container").pack(pady=5)

    src_frame = tk.Frame(win)
    src_frame.pack()
    tk.Label(src_frame, text="Host Source:").pack(side=tk.LEFT)
    host_src_entry = tk.Entry(src_frame, width=50)
    host_src_entry.pack(side=tk.LEFT)
    tk.Button(src_frame, text="Browse", command=browse_host_src).pack(side=tk.LEFT)

    dest_frame = tk.Frame(win)
    dest_frame.pack()
    tk.Label(dest_frame, text="Container Destination:").pack(side=tk.LEFT)
    container_dest_entry = tk.Entry(dest_frame, width=50)
    container_dest_entry.pack(side=tk.LEFT)
    tk.Button(dest_frame, text="Pick", command=lambda: browse_container_path(container_dest_entry)).pack(side=tk.LEFT)

    tk.Button(win, text="Copy →", command=host_to_container).pack(pady=5)

    # Container → Host
    tk.Label(win, text="Container → Host").pack(pady=10)

    c_src_frame = tk.Frame(win)
    c_src_frame.pack()
    tk.Label(c_src_frame, text="Container Source:").pack(side=tk.LEFT)
    container_src_entry = tk.Entry(c_src_frame, width=50)
    container_src_entry.pack(side=tk.LEFT)
    tk.Button(c_src_frame, text="Pick", command=lambda: browse_container_path(container_src_entry)).pack(side=tk.LEFT)

    h_dest_frame = tk.Frame(win)
    h_dest_frame.pack()
    tk.Label(h_dest_frame, text="Host Destination:").pack(side=tk.LEFT)
    host_dest_entry = tk.Entry(h_dest_frame, width=50)
    host_dest_entry.pack(side=tk.LEFT)
    tk.Button(h_dest_frame, text="Browse", command=browse_host_dest).pack(side=tk.LEFT)

    tk.Button(win, text="Copy ←", command=container_to_host).pack(pady=5)


def attach_container(name):
    container_name = resolve_container_name(name)
    if not container_name:
        messagebox.showerror("Error", f"No container found for {name}")
        return
    try:
        subprocess.Popen([
            "gnome-terminal", "--",
            "docker", "exec", "-it", container_name, "bash"
        ])
    except Exception as e:
        messagebox.showerror("Error", f"Attach failed for {container_name}:\n{e}")


def exec_command(name):
    def run_cmd():
        cmd = cmd_entry.get()
        container_name = resolve_container_name(name)
        if not container_name:
            messagebox.showerror("Error", f"No container found for {name}")
            return
        subprocess.run(["docker", "exec", container_name, "bash", "-c", cmd])

    win = Toplevel(root)
    win.title(f"Exec Command: {name}")
    tk.Label(win, text="Command:").pack()
    cmd_entry = tk.Entry(win, width=50)
    cmd_entry.pack()
    tk.Button(win, text="Run", command=run_cmd).pack()

def show_ports(name):
    container_name = resolve_container_name(name)
    try:
        output = subprocess.check_output(["docker", "port", container_name], text=True)
        messagebox.showinfo("Ports", output.strip())
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to get ports for {name}:\n{e}")

def show_stats():
    win = Toplevel(root)
    win.title("Live Container Stats")
    log_box = scrolledtext.ScrolledText(win, width=120, height=30)
    log_box.pack()

    def stats_loop():
        while True:
            try:
                # Grab all known container names including dynamic goal ones
                containers = [resolve_container_name(name) for name in container_names]

                # Extend with actual docker-goal-run-* if present
                dynamic_goals = [
                    c.name for c in client.containers.list(all=True)
                    if c.name.startswith("docker-goal-run")
                ]
                containers.extend(dynamic_goals)

                proc = subprocess.Popen(["docker", "stats", "--no-stream"] + containers,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                out, _ = proc.communicate(timeout=5)
                log_box.delete("1.0", tk.END)
                log_box.insert(tk.END, out.decode("utf-8"))
                time.sleep(3)
            except Exception as e:
                log_box.insert(tk.END, f"\nError: {e}\n")
                break

    Thread(target=stats_loop, daemon=True).start()



def scale_service(name):
    def run_scale():
        count = scale_entry.get()
        subprocess.run(["docker", "compose", "up",  "--remove-orphans", "--scale", f"{service_map[name]}={count}", "-d"])
        win.destroy()

    win = Toplevel(root)
    win.title(f"Scale Service: {name}")
    tk.Label(win, text="Instance count:").pack()
    scale_entry = tk.Entry(win, width=10)
    scale_entry.insert(0, "2")
    scale_entry.pack()
    tk.Button(win, text="Scale", command=run_scale).pack()

def up_container(name):
    try:
        if name == "goal":
            # Restart dynamic goal container using previous env vars or fallback
            subprocess.run([
                "docker", "compose", "run", "-d", "--remove-orphans", "--no-deps",
                "-e", f"GOAL_X={entry_x.get()}",
                "-e", f"GOAL_Y={entry_y.get()}",
                "-e", f"GOAL_Z={entry_z.get()}",
                "goal"
            ], check=True)
        else:
            subprocess.run(["docker", "compose", "up", "--remove-orphans", "-d", service_map[name]], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Up failed for {name}:\n{e}")


def up_all():
    try:
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
        # messagebox.showinfo("Up", "All containers started.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Up all failed:\n{e}")

def down_all():
    try:
        subprocess.run(["docker", "compose", "down"], check=True)
        # messagebox.showinfo("Down", "All containers stopped and removed.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Down all failed:\n{e}")

def down_container(name):
    try:
        if name == "goal":
            # Stop and remove all docker-goal containers
            for c in client.containers.list(all=True):
                if c.name.startswith("docker-goal"):
                    try:
                        c.stop()
                        c.remove(force=True)
                    except Exception as e:
                        print(f"Failed to remove goal container: {e}")
        else:
            subprocess.run(["docker", "compose", "rm", "-f", "-s", "-v", service_map[name]], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Down failed for {name}:\n{e}")


def find_goal_container():
    for c in client.containers.list(all=True):
        if c.name.startswith("docker-goal"):
            return c
    return None

def get_status(name):
    try:
        if name == "goal":
            # Match the most recent docker-goal-* or docker-goal-run-* container
            goal_containers = [
                c for c in client.containers.list(all=True)
                if c.name.startswith("docker-goal")
            ]
            if not goal_containers:
                return 'black', '⚫ not found'
            # Prefer running container
            for c in goal_containers:
                if c.status == 'running':
                    return 'green', '🟢 running'
            return 'yellow', '🟡 stopped'
        else:
            container = client.containers.get(name)
            if container.status == 'running':
                return 'green', '🟢 running'
            else:
                return 'yellow', '🟡 stopped'
    except docker.errors.NotFound:
        try:
            image = client.images.get(f"{name}:latest")
            return 'white', '⚪ built'
        except docker.errors.ImageNotFound:
            return 'black', '⚫ deleted'
    except Exception:
        return 'black', '⚫ error'


def update_statuses():
    for name, label in status_labels.items():
        color, text = get_status(name)
        label.config(bg=color, text=text, width=10)
    root.after(3000, update_statuses)


def start_container(name):
    try:
        container_name = resolve_container_name(name)

        subprocess.run(["docker", "compose", "start", service_map[name]], check=True)
        # messagebox.showinfo("Start", f"{name} started successfully.")
        status_labels[name].config(text=get_status(name))
    except Exception as e:
        messagebox.showerror("Error", f"Start failed for {name}:\n{e}")

def stop_container(name):
    try:

        container_name = resolve_container_name(name)
        subprocess.run(["docker", "compose", "stop" ,service_map[name]], check=True)
        # messagebox.showinfo("Stop", f"{name} stopped successfully.")
        status_labels[name].config(text=get_status(name))

    except Exception as e:
        messagebox.showerror("Error", f"Stop failed for {name}:\n{e}")

def restart_container(name):
    try:
        container_name = resolve_container_name(name)
        subprocess.run(["docker", "compose", "restart", service_map[name]], check=True)
        # messagebox.showinfo("Restart", f"{name} restarted successfully.")
        status_labels[name].config(text=get_status(name))
    except Exception as e:
        messagebox.showerror("Error", f"Restart failed for {name}:\n{e}")

def build_container(name):
    try:
        subprocess.run(["docker", "compose", "build", service_map[name]], check=True)
        # messagebox.showinfo("Build", f"{name} built successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Build failed:\n{e}")

def start_all():
    for name in container_names:
        start_container(name)

def stop_all():
    for name in container_names:
        stop_container(name)

def build_all():
    try:
        subprocess.run(["docker", "compose", "build"], check=True)
        # messagebox.showinfo("Build", "Docker Compose build completed.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Build failed:\n{e}")

def compose_down():
    try:
        subprocess.run(["docker", "compose", "down",  "--remove-orphans"], check=True)
        # messagebox.showinfo("Down", "All containers removed.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Down failed:\n{e}")

def show_logs(name):
    log_win = Toplevel(root)
    log_win.title(f"Logs: {name}")
    log_box = scrolledtext.ScrolledText(log_win, width=100, height=30)
    log_box.pack()

    def stream_logs():
        try:
            container = None
            if name == "goal":
                # Find latest dynamic goal container
                goal_containers = [
                    c for c in client.containers.list(all=True)
                    if c.name.startswith("docker-goal")
                ]
                if goal_containers:
                    # Sort by creation time (most recent first)
                    goal_containers.sort(key=lambda x: x.attrs["Created"], reverse=True)
                    container = goal_containers[0]
                else:
                    log_box.insert(tk.END, "No active 'goal' container found.\n")
                    return
            else:
                container = client.containers.get(name)

            for line in container.logs(stream=True, follow=True):
                log_box.insert(tk.END, line.decode('utf-8'))
                log_box.yview(tk.END)

        except Exception as e:
            log_box.insert(tk.END, f"\nError fetching logs:\n{e}")

    Thread(target=stream_logs, daemon=True).start()

def restart_goal_custom():
    x = entry_x.get()
    y = entry_y.get()
    z = entry_z.get()

    try:
        float(x)
        float(y)
        float(z)
    except ValueError:
        messagebox.showerror("Error", "Coordinates must be valid numbers.")
        return

    goal_service = services.get("goal")
    if not goal_service:
        messagebox.showerror("Error", "No 'goal' service found in compose file.")
        return

    # Clean up all old dynamic goal containers
    for c in client.containers.list(all=True):
        if c.name.startswith("docker-goal"):
            try:
                c.stop()
                c.remove()
            except Exception:
                pass

    try:
        subprocess.run([
            "docker", "compose", "run", "-d", "--remove-orphans", "--no-deps",
            "-e", f"GOAL_X={x}",
            "-e", f"GOAL_Y={y}",
            "-e", f"GOAL_Z={z}",
            "goal"
        ], check=True)
        # messagebox.showinfo("Success", f"Goal restarted with ({x}, {y}, {z})")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to restart goal container:\n{e}")


# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("ROS Docker Compose Manager")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

tk.Label(frame, text="ROS Docker Compose Manager", font=("Arial", 16)).grid(row=0, column=0, columnspan=8, pady=(0, 20))

# Global controls
tk.Button(frame, text="Build All", command=build_all, width=12).grid(row=1, column=0, padx=5, pady=5)
tk.Button(frame, text="Start All", command=start_all, width=12).grid(row=1, column=1, padx=5, pady=5)
tk.Button(frame, text="Stop All", command=stop_all, width=12).grid(row=1, column=2, padx=5, pady=5)
tk.Button(frame, text="Up All", command=up_all, width=12).grid(row=1, column=3, padx=5, pady=5)
tk.Button(frame, text="Down All", command=down_all, width=12).grid(row=1, column=4, padx=5, pady=5)
tk.Button(frame, text="Stats", command=show_stats, width=12).grid(row=1, column=5, padx=5, pady=5)

# Individual container controls
row_start = 2
for i, name in enumerate(container_names):
    tk.Label(frame, text=name).grid(row=row_start + i, column=0, sticky="w", pady=5)

    status = tk.Label(frame, text=get_status(name), font=("Arial", 12))
    status.grid(row=row_start + i, column=1)
    status_labels[name] = status

    tk.Button(frame, text="Start", command=lambda n=name: start_container(n)).grid(row=row_start + i, column=2)
    tk.Button(frame, text="Stop", command=lambda n=name: stop_container(n)).grid(row=row_start + i, column=3)
    tk.Button(frame, text="Restart", command=lambda n=name: restart_container(n)).grid(row=row_start + i, column=4)
    tk.Button(frame, text="Logs", command=lambda n=name: show_logs(n)).grid(row=row_start + i, column=5)
    tk.Button(frame, text="Build", command=lambda n=name: build_container(n)).grid(row=row_start + i, column=6)
    tk.Button(frame, text="Up", command=lambda n=name: up_container(n)).grid(row=row_start + i, column=7)
    tk.Button(frame, text="Down", command=lambda n=name: down_container(n)).grid(row=row_start + i, column=8)
    tk.Button(frame, text="Attach", command=lambda n=name: attach_container(n)).grid(row=row_start + i, column=9)
    tk.Button(frame, text="Exec", command=lambda n=name: exec_command(n)).grid(row=row_start + i, column=10)
    tk.Button(frame, text="Copy", command=lambda n=name: copy_files(n)).grid(row=row_start + i, column=11)
    tk.Button(frame, text="Ports", command=lambda n=name: show_ports(n)).grid(row=row_start + i, column=12)
    if any(keyword in name for keyword in scalable_keywords):
        tk.Button(frame, text="Scale", command=lambda n=name: scale_service(n)).grid(row=row_start + i, column=13)

# Coordinates input for goal_publisher
tk.Label(frame, text="New Goal Coordinates:").grid(row=row_start + len(container_names), column=0, columnspan=4, pady=(20, 5))

entry_x = tk.Entry(frame, width=10)
entry_y = tk.Entry(frame, width=10)
entry_z = tk.Entry(frame, width=10)
entry_x.insert(0, "88")
entry_y.insert(0, "-1")
entry_z.insert(0, "1")

entry_x.grid(row=row_start + len(container_names) + 1, column=0, padx=5)
entry_y.grid(row=row_start + len(container_names) + 1, column=1, padx=5)
entry_z.grid(row=row_start + len(container_names) + 1, column=2, padx=5)

tk.Button(frame, text="Restart Goal Publisher", command=restart_goal_custom, width=25).grid(
    row=row_start + len(container_names) + 2, column=0, columnspan=4, pady=10
)

update_statuses()
root.mainloop()
