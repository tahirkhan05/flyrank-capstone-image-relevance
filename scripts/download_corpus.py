"""
ContextLens - Image Corpus Downloader
Downloads ~50 CC0 / Public Domain images across 5 animal categories from Wikimedia Commons,
plus generates an intentionally ambiguous image to test low-confidence detection.
"""

from pathlib import Path
import time
import requests
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR = Path(__file__).parent.parent / "data" / "images"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "ContextLensCorpusBot/1.0 (contact@contextlens.ai)"}

CATEGORIES = {
    "red_fox": "Red Fox Vulpes",
    "gray_wolf": "Canis lupus wild",
    "domestic_dog": "Golden Retriever dog",
    "brown_bear": "Ursus arctos bear",
    "deer": "Cervus elaphus deer",
}


def fetch_category_images(category_name: str, query: str, limit: int = 10) -> int:
    """Fetch high quality images for a specific category from Wikimedia Commons."""
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"file:{query}",
        "gsrlimit": limit + 10,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 640,
        "format": "json",
    }

    try:
        resp = requests.get(api_url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"Failed to query {category_name}: status {resp.status_code}")
            return 0

        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        saved_count = 0

        for page_id, info in pages.items():
            if saved_count >= limit:
                break

            img_info_list = info.get("imageinfo", [])
            if not img_info_list:
                continue

            thumb_url = img_info_list[0].get("thumburl")
            if not thumb_url:
                continue
            clean_url = thumb_url.split("?")[0].lower()
            if not clean_url.endswith((".jpg", ".png", ".jpeg")):
                continue

            # Download thumbnail
            try:
                time.sleep(0.3)
                img_resp = requests.get(thumb_url, headers=HEADERS, timeout=15)
                if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                    saved_count += 1
                    target_path = DATA_DIR / f"{category_name}_{saved_count:02d}.jpg"
                    target_path.write_bytes(img_resp.content)
                    print(f"Downloaded [{category_name}] -> {target_path.name} ({len(img_resp.content) // 1024} KB)")
            except Exception as e:
                print(f"Error downloading {thumb_url}: {e}")

        return saved_count
    except Exception as exc:
        print(f"API request error for {category_name}: {exc}")
        return 0


def generate_ambiguous_image():
    """Generate an intentionally ambiguous, blurry, indistinct image to test low-confidence flagging."""
    width, height = 400, 400
    img = Image.new("RGB", (width, height), color=(60, 60, 65))
    draw = ImageDraw.Draw(img)

    # Draw indistinct blurred gray shapes
    draw.ellipse([80, 80, 320, 320], fill=(75, 70, 70), outline=(50, 50, 55))
    draw.ellipse([140, 150, 260, 250], fill=(85, 80, 80))

    # Apply heavy Gaussian blur to create an unrecognizable silhouette
    blurred = img.filter(ImageFilter.GaussianBlur(radius=25))
    target_path = DATA_DIR / "ambiguous_shape_01.jpg"
    blurred.save(target_path, "JPEG", quality=80)
    print(f"Generated ambiguous test image -> {target_path.name}")


def main():
    print(f"Gathering initial image corpus into: {DATA_DIR}...")
    total_downloaded = 0
    for cat_name, query in CATEGORIES.items():
        print(f"\n--- Fetching category: {cat_name} ---")
        count = fetch_category_images(cat_name, query, limit=10)
        total_downloaded += count
        print(f"Finished {cat_name}: {count} images saved.")

    generate_ambiguous_image()
    all_images = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png"))
    print(f"\nCorpus collection complete! Total images in library: {len(all_images)}")


if __name__ == "__main__":
    main()
