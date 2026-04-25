from PIL import Image
import os

BASE = "data/processed"
bad_files = []

for root, _, files in os.walk(BASE):
    for f in files:
        path = os.path.join(root, f)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            bad_files.append(path)

print("Corrupted:", len(bad_files))
for f in bad_files[:10]:
    print(f)

import hashlib
import os
from collections import defaultdict

BASE = "data/processed"
hash_map = defaultdict(list)

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

for root, _, files in os.walk(BASE):
    for f in files:
        path = os.path.join(root, f)
        h = file_hash(path)
        hash_map[h].append(path)

duplicates = [v for v in hash_map.values() if len(v) > 1]

print("Duplicate groups:", len(duplicates))

for group in duplicates:
    # keep first, remove rest
    for p in group[1:]:
        os.remove(p)
print("Removed duplicates.")


def collect_hashes(split):
    hashes = set()
    for root, _, files in os.walk(f"{BASE}/{split}"):
        for f in files:
            path = os.path.join(root, f)
            hashes.add(file_hash(path))
    return hashes

train_h = collect_hashes("train")
val_h   = collect_hashes("val")
test_h  = collect_hashes("test")

print("Train ∩ Val:", len(train_h & val_h))
print("Train ∩ Test:", len(train_h & test_h))
print("Val ∩ Test:", len(val_h & test_h))

def collect_hashes(split):
    hashes = set()
    for root, _, files in os.walk(f"{BASE}/{split}"):
        for f in files:
            path = os.path.join(root, f)
            hashes.add(file_hash(path))
    return hashes

train_h = collect_hashes("train")
val_h   = collect_hashes("val")
test_h  = collect_hashes("test")

print("Train ∩ Val:", len(train_h & val_h))
print("Train ∩ Test:", len(train_h & test_h))
print("Val ∩ Test:", len(val_h & test_h))

from collections import Counter

def count_split(split):
    counts = Counter()
    for cls in os.listdir(f"{BASE}/{split}"):
        cls_path = f"{BASE}/{split}/{cls}"
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))
    return counts

for split in ["train","val","test"]:
    print(split, count_split(split))

from PIL import Image
sizes = set()

for root, _, files in os.walk(BASE):
    for f in files[:1000]:  # sample
        path = os.path.join(root, f)
        try:
            with Image.open(path) as img:
                sizes.add(img.size)
        except:
            pass

print("Unique sizes:", sizes)


from PIL import Image

modes = set()

for root, _, files in os.walk(BASE):
    for f in files[:1000]:
        path = os.path.join(root, f)
        try:
            with Image.open(path) as img:
                modes.add(img.mode)
        except:
            pass

print("Modes:", modes)

# normalize_images.py
import os
from PIL import Image

BASE = "data/processed"
TARGET_SIZE = (224, 224)  # or (512, 512)

for root, _, files in os.walk(BASE):
    for f in files:
        path = os.path.join(root, f)

        try:
            img = Image.open(path)

            # remove alpha if exists
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # convert to grayscale
            img = img.convert("L")

            # resize
            img = img.resize(TARGET_SIZE)

            # overwrite as PNG
            new_path = os.path.splitext(path)[0] + ".png"
            img.save(new_path)

            if new_path != path:
                os.remove(path)

        except Exception:
            os.remove(path)

print("Normalization done.")