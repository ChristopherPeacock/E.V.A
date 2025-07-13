from tools import nmap, ping


def test_nmap_accepts_target():
    result = nmap.run_nmap("127.0.0.1") 
    assert "nmap scan results" in result.lower() 


def test_can_ping_target():
    result = ping.run_ping("google.com")
    assert "is reachable" in result.lower()