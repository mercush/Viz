import subprocess

def convert_and_open():
    subprocess.run(["vl2png", "temp.json", "temp.png"])
    subprocess.run(["open", "temp.png"])
