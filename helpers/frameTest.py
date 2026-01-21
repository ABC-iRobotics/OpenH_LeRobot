import os
from datetime import datetime
import matplotlib.pyplot as plt

BASE = "/home/lorant/OpenH_LeRobot/recordings/20260116_111138"

paths = {
    "decklink0": BASE + "/decklink0",
    "decklink1": BASE + "/decklink1",
    "usb8": BASE + "/usb8",
    "usb6": BASE + "/usb6",
}


start = 1
end = 20001

frames = []
diff_decklink1 = []
diff_usb8 = []
diff_usb6 = []

'''
for i in range(101,180):
    print(os.stat(os.path.join(paths["decklink0"], f"frame_000{i}.jpg")).st_mtime_ns/1e9 - 
          os.stat(os.path.join(paths["decklink0"], f"frame_000{i-1}.jpg")).st_mtime_ns/1e9)
'''



#print(os.stat(os.path.join(paths["decklink0"], "frame_000075.jpg")).st_ctime_ns/1e9)
print(os.stat(os.path.join(paths["decklink0"], "frame_000471.jpg")).st_mtime_ns/1e9)
print()
print(os.stat(os.path.join(paths["decklink1"], "frame_000471.jpg")).st_mtime_ns/1e9)
print()
print(os.stat(os.path.join(paths["usb6"], "frame_000250.jpg")).st_mtime_ns/1e9)
print()
print(os.stat(os.path.join(paths["usb8"], "frame_000249.jpg")).st_mtime_ns/1e9)

#print(os.stat(os.path.join(paths["decklink1"], "frame_000075.jpg")).st_mtime_ns/1e9)
#print(os.stat(os.path.join(paths["usb6"], "frame_000075.jpg")).st_mtime_ns/1e9)
#print(os.stat(os.path.join(paths["usb8"], "frame_000075.jpg")).st_mtime_ns/1e9)
'''

for i in range(start, end + 1):
    fname = f"frame_{i:06d}.jpg"

    try:
        s0 = os.stat(os.path.join(paths["decklink0"], fname))
        s1 = os.stat(os.path.join(paths["decklink1"], fname))
        s2 = os.stat(os.path.join(paths["usb8"], fname))
        s3 = os.stat(os.path.join(paths["usb6"], fname))
    except FileNotFoundError:
        continue

    # nanosecond precision → seconds
    t0 = s0.st_mtime_ns / 1e9
    t1 = s1.st_mtime_ns / 1e9
    t2 = s2.st_mtime_ns / 1e9
    t3 = s3.st_mtime_ns / 1e9

    frames.append(i)
    diff_decklink1.append(t1 - t0)
    diff_usb8.append(t2 - t0)
    diff_usb6.append(t3 - t0)

# Plot
plt.figure()
plt.plot(frames, diff_decklink1, label="decklink1 - decklink0")
plt.plot(frames, diff_usb8, label="usb8 - decklink0")
plt.plot(frames, diff_usb6, label="usb6 - decklink0")

plt.xlabel("Frame index")
plt.ylabel("Creation time difference (seconds)")
plt.title("Per-frame creation-time differences")
plt.legend()
plt.grid(True)

plt.show()
'''
