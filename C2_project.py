# first things first lets get the data we ned for the tasl
import cv2
from pathlib import Path

data_dir = Path("data_provided_for_task/IconDataset/png")
for p in data_dir.rglob("*.png"):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA if has alpha
    # process as numpy array