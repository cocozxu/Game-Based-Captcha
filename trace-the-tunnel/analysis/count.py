import os

folder = "trace-the-tunnel/data/human"  # change this
count = len([
    f for f in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, f))
])

print("Number of files:", count)
