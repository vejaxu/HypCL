"""Extract downloaded paper splits and validate them with the project loader."""
import json
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from utils.data import load_dataset

DATASETS = {
    "ucf101": ("ucf.zip", ""),
    "cub": ("cub.zip", ""),
    "imagenetr": ("imagenetr.zip", "imagenetr"),
    "cars": ("cars.zip", ""),
    "aircraft": ("aircraft.tar", ""),
    "food101": ("food101.tar", ""),
    "sun": ("sun.tar", ""),
    "objectnet": ("objectnet.tar.gz", ""),
}


def prepare(name):
    if name != "cifar100":
        filename, subdir = DATASETS[name]
        archive = ROOT / "data/downloads" / filename
        marker = archive.with_name(archive.name + ".complete.json")
        if not marker.is_file():
            raise FileNotFoundError(f"Download not yet complete: {archive}")
        destination = ROOT / "data" / subdir
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {name}", flush=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zipped:
                for info in zipped.infolist():
                    path = (destination / info.filename).resolve()
                    if not path.is_relative_to(destination.resolve()):
                        raise ValueError(f"Unsafe archive member: {info.filename}")
                    if (info.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError(f"Archive contains symlink: {info.filename}")
                zipped.extractall(destination)
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(destination, filter="data")
    images, labels, paths = load_dataset(name, ROOT / "data")
    if paths:
        # Decode a sample from every class in each split.
        import numpy as np
        for split in images:
            for label in np.unique(labels[split]):
                sample = images[split][np.flatnonzero(labels[split] == label)[0]]
                with Image.open(sample) as image:
                    image.convert("RGB").load()
    result = {split: {"samples": len(labels[split]),
                       "classes": len(set(labels[split].tolist()))}
              for split in labels}
    output = ROOT / "data" / f"{name}.prepared.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {name}: {result}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        prepare(name)
