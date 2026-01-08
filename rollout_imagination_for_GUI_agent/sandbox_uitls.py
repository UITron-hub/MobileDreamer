import threading
from pathlib import Path
import json
from trajectory_rollout.client import test_address_available

IP_POOL = []

LOCK = threading.Lock()


def load_existing_sandboxes():
    ip_cache_file = Path(__file__).parent / "online_machine_cache.json"
    if ip_cache_file.exists():
        try:
            with open(ip_cache_file, "r") as f:
                ip_cache = json.load(f)
        except:
            ip_cache = []
        for host_ip, host_port in ip_cache:
            if test_address_available(host_ip, host_port):
                IP_POOL.append((host_ip, host_port))
        with open(ip_cache_file, "w") as f:
            json.dump(IP_POOL, f, indent=2)


def add_sandbox_to_log(host_ip, host_port):
    ip_cache_file = Path(__file__).parent / "online_machine_cache.json"
    with LOCK:
        try:
            with open(ip_cache_file, "r") as f:
                ip_cache = json.load(f)
        except:
            ip_cache = []

        if [host_ip, host_port] not in ip_cache:
            ip_cache.append([host_ip, host_port])
            with open(ip_cache_file, "w") as f:
                json.dump(ip_cache, f, indent=2)


def remove_host_from_log(host_ip, host_port):
    ip_cache_file = Path(__file__).parent / "online_machine_cache.json"
    with LOCK:
        if not ip_cache_file.exists():
            return
        with open(ip_cache_file, "r") as f:
            ip_cache = json.load(f)
        if [host_ip, host_port] in ip_cache:
            ip_cache.remove([host_ip, host_port])
            with open(ip_cache_file, "w") as f:
                json.dump(ip_cache, f, indent=2)
