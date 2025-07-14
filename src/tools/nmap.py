import subprocess
import shutil

def run_nmap(target: str) -> str:
    """Run nmap scan on a target. Input should be an IP address or hostname."""

    if not shutil.which("nmap"):
        return """Nmap is not installed on this system. 
            To install:
                - Windows: Download from https://nmap.org/download.html
                - Linux: sudo apt install nmap (Ubuntu/Debian) or sudo yum install nmap (CentOS/RHEL)
                - macOS: brew install nmap

            Alternative: I can use ping to check host connectivity."""
    try:
        result = subprocess.run(
            ["nmap", "-sn", target], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        if result.returncode == 0:
            return f"Nmap scan results for {target}: \n{result.stdout}"
        else:
            return f"Nmap scan failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Nmap scan of {target} timed out"
    except Exception as e:
        return f"Error running nmap on {target}: {str(e)}"
