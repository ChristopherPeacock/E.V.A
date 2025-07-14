import subprocess
import os

def run_ping(target: str) -> str:
    """Ping a target host. Input should be an IP address or hostname."""
    try:
        ping_cmd = ["ping", "-n", "4", target] if os.name == 'nt' else ["ping", "-c", "4", target]
        result = subprocess.run(ping_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return f" {target} is reachable:\n{result.stdout}"
        else:
            return f" {target} is not reachable:\n{result.stderr}"
    except Exception as e:
        return f"Error pinging {target}: {str(e)}"
