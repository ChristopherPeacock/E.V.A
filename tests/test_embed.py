from utils.embed import run_nmap
from utils.embed import run_ping

def test_nmap_accepts_target():
    result = run_nmap("127.0.0.1") 
    assert "nmap scan results" in result.lower() 


def test_can_ping_target():
    result = run_ping("google.com")
    assert "is reachable" in result.lower()