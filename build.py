import subprocess
import os 

command = f"rm -rf build && rm -rf dist && rm -rf kiwigrad.egg-info && python3 setup.py sdist bdist_wheel"
subprocess.run(command, shell=True)