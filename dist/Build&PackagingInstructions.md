# Build & Packaging Instructions

This guide helps you rebuild the AI CLI executable with updated name, icon, or other options.

---

## 1. Install PyInstaller (if not installed)

```bash
pip install pyinstaller
```

## 2. Build Executable with Custom Name and Icon

```bash
pyinstaller --onefile --console --name adam_assistant --icon assets/adam_icon.ico main.py
```
- onefile packages everything into a single .exe

- console shows terminal window (use --windowed for no console)

- name sets the output executable name

- icon sets the custom .ico file for the executable

# 3. Using a .spec file (optional, for bundling assets)

- Edit main.spec to update:

```bash 
exe = EXE(
    ...
    name='adam_assistant.exe',
    icon='assets/adam_icon.ico',
    ...
)
```

- rebuild with

```bash
pyinstaller main.spec
```

## Notes 

- Make sure your icon file is a valid .ico with multiple sizes.

- After building, check dist/ folder for the .exe.

- If the icon does not update in Windows Explorer, try restarting Explorer or clearing icon cache.

## Optional

- To update version info in the app:

```bash
__version__ = "v1.0.5"
```

- Import and print version in main.py:

```bash
from version import __version__
print(f"A.D.A.M {__version__} online.")
```

## Running the Executable

- Double-click dist/adam_assistant.exe to launch.

```bash
I want to automate this further with scripts or GitHub Actions workflows soon.
```
