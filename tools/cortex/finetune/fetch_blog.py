__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import requests
from config import BLOG_OUTPUT_DIR, BLOG_SITEMAP_URL
from run_helper import banner

logger = logging.getLogger(__name__)


def fetch_post_urls(sitemap_url):
    response = requests.get(sitemap_url, timeout=30)
    response.raise_for_status()
    urls = re.findall(r"<loc>(https?://[^<]+)</loc>", response.text)
    # Exclude the /blog/ index page itself
    return [u for u in urls if not u.rstrip("/").endswith("/blog")]


def extract_post_text(html):
    """Extract the main post body text from WordPress HTML, stripping nav/sidebar/comments."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup.select("nav, header, footer, aside, .comments-area, .comment-respond, "
                            ".wp-block-navigation, .site-header, .site-footer, script, style"):
        tag.decompose()

    # Try WordPress standard content containers
    for selector in (".entry-content", ".post-content", "article", ".post"):
        content = soup.select_one(selector)
        if content:
            return content.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


def slugify(url):
    """Convert a URL to a safe filename slug."""
    path = url.rstrip("/").split("//", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-")
    return slug[:120]


def fetch_blog(output_dir, sitemap_url, delay, limit):
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching post URLs from {sitemap_url} ...")
    urls = fetch_post_urls(sitemap_url)
    logger.info(f"Found {len(urls)} posts")

    if limit:
        urls = urls[:limit]

    written, skipped = 0, 0
    for url in urls:
        slug = slugify(url)
        out_path = output_dir / f"{slug}.txt"

        if out_path.exists():
            logger.info(f"  Skipping (exists): {slug}")
            skipped += 1
            continue

        try:
            logger.info(f"  Fetching: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = extract_post_text(response.text)
            if text.strip():
                out_path.write_text(text, encoding="utf-8")
                written += 1
            else:
                logger.warning(f"  Empty content for {url} — skipping")
                skipped += 1
        except Exception as e:
            logger.warning(f"  Failed to fetch {url}: {e}")
            skipped += 1

        if delay:
            time.sleep(delay)

    logger.info(f"Done — {written} posts written, {skipped} skipped → {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fetch blog posts as plain text for CPT tone training.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for .txt files")
    parser.add_argument("--sitemap", default=BLOG_SITEMAP_URL, help="WordPress post sitemap URL")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default: 1.0)")
    parser.add_argument("--limit", type=int, default=0, help="Max posts to fetch (0 = all)")
    args = parser.parse_args()

    output_dir = args.output or BLOG_OUTPUT_DIR

    banner("FETCH-BLOG — STARTING")
    fetch_blog(
        output_dir=output_dir,
        sitemap_url=args.sitemap,
        delay=args.delay,
        limit=args.limit,
    )
    banner("FETCH-BLOG — DONE")
