"""
download_books.py

Downloads the sample corpus used by the Cosmos DB RAG tutorial.

The source documents are public-domain books available through
Internet Archive / Project Gutenberg.
"""

from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent
DATA_DIR.mkdir(exist_ok=True)

PDFS = {
    "ancient_egyptian_costumes.pdf":
        "https://ia601605.us.archive.org/view_archive.php?archive=/33/items/GutenbergENzip/01.zip&file=Ancient%20Egyptian%2C%20Assyrian%2C%20and%20Persian%20costumes%20and%20decorations%20-%20Florence%20S.%20Hornblower%20%26%20Mary%20G.%20Houston%2C%202017%20%2856p%29.pdf",

    "egyptian_immortality.pdf":
        "https://ia601605.us.archive.org/view_archive.php?archive=/33/items/GutenbergENzip/01.zip&file=Ancient%20Egyptian%20Doctrine%20of%20the%20Immortality%20of%20the%20Soul%2C%20The%20-%20Alfred%20Wiedemann%2C%202015%20%2835p%29.pdf",

    "egyptian_greek_looms.pdf":
        "https://ia601605.us.archive.org/view_archive.php?archive=/33/items/GutenbergENzip/01.zip&file=Ancient%20Egyptian%20and%20Greek%20Looms%20-%20H.%20Ling%20Roth%2C%202008%20%2849p%29.pdf",

    "ancient_rome_history.pdf":
        "https://ia601605.us.archive.org/view_archive.php?archive=/33/items/GutenbergENzip/01.zip&file=Ancient%20Rome%20_%20from%20the%20earliest%20times%20down%20to%20476%20A.%20D.%20-%20Robert%20F.%20Pennell%2C%202004%20%28123p%29.pdf",

    "ancient_history_volume_1.pdf":
        "https://ia601605.us.archive.org/view_archive.php?archive=/33/items/GutenbergENzip/01.zip&file=Ancient%20History%20of%20the%20Egyptians%2C%20Carthaginians%2C%20Assyrians%2C%20_%20B%20and%20Grecians%20_%20%28Vol.%201%20of%206%29%2C%20The%20-%20Charles%20Rollin%2C%202009%20%28312p%29.pdf",
}


def download_file(url: str, output_path: Path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


for filename, url in PDFS.items():
    destination = DATA_DIR / filename

    if destination.exists():
        print(f"Skipping {filename}")
        continue

    print(f"Downloading {filename}")
    download_file(url, destination)

print("\nDownload complete.")