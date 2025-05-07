import csv

import matplotlib.pyplot as plt

f = open("single.csv", mode="r", encoding="utf-8", newline="")
csv_reader = csv.DictReader(
    f,
    fieldnames=[
        "rank",
        "pid",
        "memory-rss",
        "memory-vms",
        "memory-data",
        "cpu-percent",
    ],
)
next(csv_reader)
rss_single = list()
for line in csv_reader:
    rss_single.append(float(line["memory-rss"]))
x = range(len(rss_single))
plt.figure()
plt.subplot(121)
plt.plot(x, rss_single)
plt.title("single process rss wo FSDP")
plt.xlabel("second")
plt.ylabel("rss GB")
plt.subplot(122)

f = open("fsdp_2DDP.csv", mode="r", encoding="utf-8", newline="")
csv_reader = csv.DictReader(
    f,
    fieldnames=[
        "rank",
        "pid",
        "memory-rss",
        "memory-vms",
        "memory-data",
        "cpu-percent",
    ],
)
next(csv_reader)
line = next(csv_reader)
pid = line["pid"]
rss_2ddp = list()
for line in csv_reader:
    if line["pid"] == pid:
        rss_2ddp.append(float(line["memory-rss"]))
x = range(len(rss_2ddp))
plt.plot(x, rss_2ddp)
plt.title("single process rss in FSDP")
plt.xlabel("second")
plt.ylabel("rss GB")
plt.savefig("rss.png")
