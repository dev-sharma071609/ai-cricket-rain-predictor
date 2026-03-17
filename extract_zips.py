from pathlib import Path
import zipfile

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

ZIP_FILES = {
    "t20s": BASE_DIR / "t20s_json.zip",
    "odis": BASE_DIR / "odis_json.zip",
    "tests": BASE_DIR / "tests_json.zip",
}

def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    print(f"Extracted {zip_path.name} -> {output_dir}")

def main():
    for folder_name, zip_path in ZIP_FILES.items():
        if not zip_path.exists():
            print(f"Missing zip file: {zip_path}")
            continue

        extract_zip(zip_path, RAW_DIR / folder_name)

if __name__ == "__main__":
    main()
