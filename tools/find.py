import subprocess

def find_file(command: str,) -> str:
    try:
        command = ['find', command ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            if command is None:
                return F"couldnt find {command}"
            else:
                return F"the file is: {result.stderr}"
    except Exception as e:
        return f"error using command line: {str(e)}"