"""Download the open grid network and plant list into data/raw/. Finished files are skipped."""
from pathlib import Path
from urllib.request import urlretrieve

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
ZENODO = "https://zenodo.org/api/records/18619025/files/{}/content"
FILES = {name: ZENODO.format(name) for name in ("buses.csv", "lines.csv", "transformers.csv")}
FILES["powerplants.csv"] = "https://raw.githubusercontent.com/PyPSA/powerplantmatching/master/powerplants.csv"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = RAW / name
        if not path.exists():
            urlretrieve(url, path)
            print("downloaded", name)


if __name__ == "__main__":
    main()
