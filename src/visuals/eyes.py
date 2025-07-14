import time
import os
from dotenv import load_dotenv

load_dotenv()

eye_frames = [
    r"""
     ┌──────────┐
     │  ●    ●  │
     │          │
     └──────────┘
    """,
    r"""
     ┌──────────┐
     │  ◉    ◉  │
     │          │
     └──────────┘
    """,
    r"""
     ┌──────────┐
     │  ◎    ◎  │
     │          │
     └──────────┘
    """
]

def animate_eyes(cycles=3, delay=0.2):
    try:
        for _ in range(cycles):
            for frame in eye_frames:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(frame)
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[✋] Eye animation interrupted.")

animate_eyes()