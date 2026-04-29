#!/usr/bin/env python3
"""
Douban Celebrity Crawler for KGQA Movie Knowledge Graph Expansion.

Starts from the most-connected celebrities in Top250 movies, then expands
through their complete filmographies to discover more movies beyond Top250.

Four-phase pipeline with checkpoint-based state persistence:
  Phase 1: Discover top celebrities from Top250 (no country filter)
  Phase 2: Paginate each celebrity's filmography to find new movie URLs
  Phase 3: Crawl detail pages for newly discovered movies
  Phase 4: Generate KG-compatible CSV + JSON outputs
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tools.douban_crawler import (  # noqa: E402
    KG_CSV_FIELDS,
    TOP250_URL,
    LinkParser,
    MovieRecord,
    PoliteFetcher,
    _collect_links,
    build_kg_rows,
    extract_douban_id,
    normalize_space,
    parse_movie_detail,
    parse_top250_movie_links,
    unique_keep_order,
)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CELEBRITY_LIMIT = 100
DEFAULT_MOVIE_LIMIT = 500
DEFAULT_OUTPUT_DIR = "data/celebrity"
DEFAULT_DELAY_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 15
FILMOGRAPHY_URL_TEMPLATE = (
    "https://movie.douban.com/celebrity/{douban_id}/movies"
    "?start={start}&format=pic&sortby=time"
)
TOP250_PAGE_SIZE = 25
TOP250_TOTAL = 250
CELEBRITY_FILMOGRAPHY_PAGE_SIZE = 30
MAX_RETRIES_ON_CHALLENGE = 5
CHALLENGE_BACKOFF_BASE = 60.0  # seconds
DOUBAN_HOMEPAGE = "https://movie.douban.com"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
}

CHECKPOINT_VERSION = 1


# ── Dataclass ────────────────────────────────────────────────────────────────


@dataclass
class CelebrityRef:
    """A celebrity discovered from movies, ranked by Top250 appearances."""

    name: str
    douban_id: str
    url: str
    top250_count: int = 0


# ── Parsing Utilities ────────────────────────────────────────────────────────


def parse_celebrity_filmography_page(html, base_url):
    """Extract movie links from a Douban celebrity filmography grid page.

    Returns a list of {"douban_id": str, "url": str, "title": str} dicts,
    deduplicated by douban_id in page order.
    """
    parser = LinkParser()
    parser.feed(html)

    refs = []
    seen_ids = set()
    for link in parser.links:
        href = link["href"]
        match = re.search(r"/subject/(\d+)/?", href)
        if not match:
            continue
        douban_id = match.group(1)
        if douban_id in seen_ids:
            continue
        seen_ids.add(douban_id)
        refs.append(
            {
                "douban_id": douban_id,
                "url": href.split("?")[0],
                "title": normalize_space(link["title"] or link["text"]),
            }
        )
    return refs


def is_douban_challenge_page(html):
    """Detect Douban anti-bot challenge pages."""
    if not html:
        return False
    if len(html) < 5000 and '<form name="sec"' in html:
        return True
    if "检测到有异常请求从你的" in html or "验证你为正常人" in html:
        return True
    return False


# ── Checkpoint System ────────────────────────────────────────────────────────


def _init_state():
    return {
        "version": CHECKPOINT_VERSION,
        "phase": 1,
        "top_celebrities": [],
        "celebrity_index": 0,
        "filmography_start": {},
        "discovered_movie_urls": [],
        "movie_index": 0,
        "movies": [],
        "stats": {
            "top250_movies_crawled": 0,
            "celebrities_discovered": 0,
            "filmography_pages_crawled": 0,
            "new_movies_crawled": 0,
        },
    }


def _save_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "version": state["version"],
        "phase": state["phase"],
        "top_celebrities": [asdict(c) for c in state["top_celebrities"]],
        "celebrity_index": state["celebrity_index"],
        "filmography_start": state["filmography_start"],
        "discovered_movie_urls": state["discovered_movie_urls"],
        "movie_index": state["movie_index"],
        "movies": [asdict(m) for m in state["movies"]],
        "stats": state["stats"],
    }
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_checkpoint(path):
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"⚠️  Checkpoint file corrupt, starting fresh: {path}")
        return None

    if raw.get("version") != CHECKPOINT_VERSION:
        print(f"⚠️  Checkpoint version mismatch, starting fresh: {path}")
        return None

    state = _init_state()
    state["version"] = raw.get("version", CHECKPOINT_VERSION)
    state["phase"] = raw.get("phase", 1)
    state["top_celebrities"] = [
        CelebrityRef(**c) for c in raw.get("top_celebrities", [])
    ]
    state["celebrity_index"] = raw.get("celebrity_index", 0)
    state["filmography_start"] = raw.get("filmography_start", {})
    state["discovered_movie_urls"] = raw.get("discovered_movie_urls", [])
    state["movie_index"] = raw.get("movie_index", 0)
    state["movies"] = [MovieRecord(**m) for m in raw.get("movies", [])]
    state["stats"] = raw.get("stats", state["stats"])
    return state


# ── Main Crawler Class ──────────────────────────────────────────────────────


class CelebrityCrawler:
    """Crawls Douban celebrities and their complete filmographies."""

    def __init__(
        self,
        fetcher,
        celebrity_limit=DEFAULT_CELEBRITY_LIMIT,
        movie_limit=DEFAULT_MOVIE_LIMIT,
        checkpoint_path=None,
    ):
        self.fetcher = fetcher
        self.celebrity_limit = celebrity_limit
        self.movie_limit = movie_limit
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.state = _init_state()

    def crawl(self):
        """Run the full crawl pipeline, dispatching to the appropriate phase."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            loaded = _load_checkpoint(self.checkpoint_path)
            if loaded:
                self.state = loaded
                print(
                    f"📂 Resuming from checkpoint — "
                    f"phase {self.state['phase']}, "
                    f"{len(self.state['movies'])} movies so far"
                )

        if self.state["phase"] <= 1:
            self._phase1_discover()

        if self.state["phase"] <= 2:
            self._phase2_filmography()

        if self.state["phase"] <= 3:
            self._phase3_enrich_movies()

        return self.state["movies"], self.state["top_celebrities"]

    # ── Phase 1: Discover top celebrities from Top250 ─────────────────────

    def _phase1_discover(self):
        print("━" * 50)
        print("🔍 Phase 1: Discovering top celebrities from Top250")
        print("━" * 50)

        if self.state["top_celebrities"]:
            print(
                f"  Already have {len(self.state['top_celebrities'])} "
                f"top celebrities from checkpoint, skipping Phase 1"
            )
            self.state["phase"] = 2
            self._maybe_checkpoint()
            return

        # 1a. Fetch all Top250 listing pages
        all_movie_urls = []
        for start in range(0, TOP250_TOTAL, TOP250_PAGE_SIZE):
            list_url = f"{TOP250_URL}?start={start}&filter="
            print(f"  Fetching Top250 page: start={start}")
            html = self._fetch_with_retry(list_url)
            page_urls = parse_top250_movie_links(html, list_url)
            all_movie_urls.extend(page_urls)
            if len(all_movie_urls) >= TOP250_TOTAL:
                break

        all_movie_urls = unique_keep_order(all_movie_urls)
        print(f"  Found {len(all_movie_urls)} unique movie URLs from Top250")

        # 1b. Crawl each movie detail page, collect ALL celebrity URLs
        celebrity_counter = Counter()  # douban_id -> appearance count
        celebrity_info = {}  # douban_id -> {"name": ..., "url": ...}
        movies_crawled = self.state["stats"]["top250_movies_crawled"]
        start_idx = movies_crawled

        for i, url in enumerate(all_movie_urls[start_idx:]):
            idx = start_idx + i
            print(f"  [{idx + 1}/{len(all_movie_urls)}] {url}")
            try:
                html = self._fetch_with_retry(url)
                movie = parse_movie_detail(html, url)
            except Exception as exc:
                print(f"    ⚠️  Skipping: {exc}")
                self.state["stats"]["top250_movies_crawled"] = idx + 1
                self._maybe_checkpoint()
                continue

            if not movie.title or not movie.douban_id:
                self.state["stats"]["top250_movies_crawled"] = idx + 1
                self._maybe_checkpoint()
                continue

            self.state["movies"].append(movie)

            # Extract ALL celebrity links (directors + actors) from the page
            for link in _collect_links(html, url):
                if not re.search(r"/celebrity/\d+/?", link["href"]):
                    continue
                celeb_id = extract_douban_id(link["href"])
                if not celeb_id:
                    continue
                celebrity_counter[celeb_id] += 1
                if celeb_id not in celebrity_info:
                    name = normalize_space(link["title"] or link["text"])
                    celebrity_info[celeb_id] = {
                        "name": name,
                        "url": link["href"].split("?")[0],
                    }

            self.state["stats"]["top250_movies_crawled"] = idx + 1
            if idx % 10 == 0:
                self._maybe_checkpoint()

        # 1c. Rank and select top celebrities
        top_ids = [
            cid
            for cid, _count in celebrity_counter.most_common(
                self.celebrity_limit
            )
        ]
        self.state["top_celebrities"] = [
            CelebrityRef(
                name=celebrity_info[cid]["name"],
                douban_id=cid,
                url=celebrity_info[cid]["url"],
                top250_count=celebrity_counter[cid],
            )
            for cid in top_ids
            if cid in celebrity_info
        ]

        print(
            f"  Selected top {len(self.state['top_celebrities'])} celebrities "
            f"from {len(celebrity_counter)} unique celebrities"
        )
        for i, celeb in enumerate(self.state["top_celebrities"][:10], 1):
            print(
                f"    {i:>3}. {celeb.name:<16} "
                f"(Top250 x{celeb.top250_count})"
            )

        self.state["phase"] = 2
        self._maybe_checkpoint()

    # ── Phase 2: Paginate filmography for each celebrity ─────────────────

    def _phase2_filmography(self):
        print("━" * 50)
        print("🎬 Phase 2: Discovering movies from celebrity filmographies")
        print("━" * 50)

        top_celebrities = self.state["top_celebrities"]
        celebrity_index = self.state["celebrity_index"]

        known_movie_ids = {
            m.douban_id
            for m in self.state["movies"]
            if m.douban_id
        }
        for ref in self.state["discovered_movie_urls"]:
            known_movie_ids.add(ref["douban_id"])

        for ci in range(celebrity_index, len(top_celebrities)):
            celebrity = top_celebrities[ci]
            celeb_id = celebrity.douban_id
            print(
                f"  [{ci + 1}/{len(top_celebrities)}] "
                f"{celebrity.name} (id={celeb_id}, "
                f"Top250 x{celebrity.top250_count})"
            )

            start = self.state["filmography_start"].get(celeb_id, 0)
            pages_for_celeb = 0
            consecutive_empty = 0
            new_for_this_celeb = 0

            while True:
                filmo_url = FILMOGRAPHY_URL_TEMPLATE.format(
                    douban_id=celeb_id,
                    start=start,
                )
                try:
                    html = self._fetch_with_retry(filmo_url)
                except Exception as exc:
                    print(f"    ⚠️  Error fetching filmography: {exc}")
                    break

                refs = parse_celebrity_filmography_page(html, filmo_url)
                for ref in refs:
                    if ref["douban_id"] not in known_movie_ids:
                        known_movie_ids.add(ref["douban_id"])
                        self.state["discovered_movie_urls"].append(ref)
                        new_for_this_celeb += 1

                pages_for_celeb += 1
                self.state["stats"]["filmography_pages_crawled"] += 1
                self.state["filmography_start"][celeb_id] = start

                if not refs:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break
                else:
                    consecutive_empty = 0

                start += len(refs) or CELEBRITY_FILMOGRAPHY_PAGE_SIZE

                if pages_for_celeb >= 30:
                    print(f"    (reached page cap for {celebrity.name})")
                    break

            print(
                f"    → {new_for_this_celeb} new movies, "
                f"total discovered: {len(self.state['discovered_movie_urls'])}"
            )

            self.state["celebrity_index"] = ci + 1
            self.state["filmography_start"].pop(celeb_id, None)
            self._maybe_checkpoint()

        self.state["phase"] = 3
        self._maybe_checkpoint()

    # ── Phase 3: Crawl detail pages for newly discovered movies ───────────

    def _phase3_enrich_movies(self):
        print("━" * 50)
        print("📝 Phase 3: Crawling new movie detail pages")
        print("━" * 50)

        existing_ids = {m.douban_id for m in self.state["movies"] if m.douban_id}
        movie_index = self.state["movie_index"]
        discovered = self.state["discovered_movie_urls"]
        movies_to_crawl = [
            ref for ref in discovered if ref["douban_id"] not in existing_ids
        ]

        print(f"  {len(movies_to_crawl)} new movies to crawl")

        for mi in range(
            movie_index, min(len(movies_to_crawl), self.movie_limit)
        ):
            ref = movies_to_crawl[mi]
            url = ref["url"]
            print(
                f"  [{mi + 1}/{min(len(movies_to_crawl), self.movie_limit)}] "
                f"{ref.get('title', url)}"
            )

            try:
                html = self._fetch_with_retry(url)
                movie = parse_movie_detail(html, url)
            except Exception as exc:
                print(f"    ⚠️  Error: {exc}")
                self.state["movie_index"] = mi + 1
                self._maybe_checkpoint()
                continue

            if not movie.title or not movie.douban_id:
                self.state["movie_index"] = mi + 1
                self._maybe_checkpoint()
                continue

            self.state["movies"].append(movie)
            self.state["stats"]["new_movies_crawled"] += 1
            self.state["movie_index"] = mi + 1

            if mi % 10 == 0:
                self._maybe_checkpoint()

        print(
            f"  Crawled {self.state['stats']['new_movies_crawled']} new movies"
        )
        self.state["phase"] = 4
        self._maybe_checkpoint()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _fetch_with_retry(self, url):
        """Fetch a URL with automatic retry on anti-bot challenge pages."""
        import random

        for attempt in range(1, MAX_RETRIES_ON_CHALLENGE + 1):
            html = self.fetcher.fetch(url)
            if not is_douban_challenge_page(html):
                return html

            # Exponential backoff with jitter
            wait = CHALLENGE_BACKOFF_BASE * (2 ** (attempt - 1))
            wait *= random.uniform(0.5, 1.5)
            print(
                f"    ⚠️  Anti-bot challenge, "
                f"waiting {wait:.0f}s (retry {attempt}/{MAX_RETRIES_ON_CHALLENGE})"
            )
            time.sleep(wait)

            # Delete cached challenge page so next attempt re-fetches
            cache_path = self.fetcher._cache_path(url)
            if cache_path.exists():
                cache_path.unlink()

        self._maybe_checkpoint()
        raise RuntimeError(
            f"Persistent anti-bot challenge after "
            f"{MAX_RETRIES_ON_CHALLENGE} retries for: {url}"
        )

    def _maybe_checkpoint(self):
        if self.checkpoint_path:
            try:
                _save_checkpoint(self.state, self.checkpoint_path)
            except OSError as exc:
                print(f"  ⚠️  Failed to save checkpoint: {exc}")


# ── Output Functions ────────────────────────────────────────────────────────


def write_celebrity_outputs(output_dir, movies, top_celebrities):
    """Write all output files to the given directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    movies_json = output_path / "movies.json"
    celebrities_json = output_path / "celebrities.json"
    kg_csv = output_path / "movies_data_douban.csv"
    report_json = output_path / "crawl_report.json"

    movies_json.write_text(
        json.dumps(
            [asdict(m) for m in movies], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    celebrities_json.write_text(
        json.dumps(
            [asdict(c) for c in top_celebrities],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = build_kg_rows(movies)
    with kg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KG_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "movie_count": len(movies),
        "celebrity_count": len(top_celebrities),
        "kg_row_count": len(rows),
        "files": {
            "movies": str(movies_json),
            "celebrities": str(celebrities_json),
            "kg_csv": str(kg_csv),
        },
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crawl Douban celebrities and their complete "
        "filmographies for KGQA expansion."
    )
    parser.add_argument(
        "--celebrity-limit",
        type=int,
        default=DEFAULT_CELEBRITY_LIMIT,
        help=f"Top N celebrities by Top250 appearances (default: {DEFAULT_CELEBRITY_LIMIT})",
    )
    parser.add_argument(
        "--movie-limit",
        type=int,
        default=DEFAULT_MOVIE_LIMIT,
        help=f"Max new movie detail pages to crawl in Phase 3 (default: {DEFAULT_MOVIE_LIMIT})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HTML cache directory (default: <output-dir>/cache)",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint file path (default: <output-dir>/checkpoint.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Force resume from checkpoint (auto-detected by default)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Delay between requests to same host in seconds (default: {DEFAULT_DELAY_SECONDS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--user-agent",
        default="kgqa-movie-research-crawler/1.0",
        help="User-Agent header",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    )
    checkpoint_path = (
        Path(args.checkpoint_file)
        if args.checkpoint_file
        else output_dir / "checkpoint.json"
    )

    fetcher = PoliteFetcher(
        cache_dir=cache_dir,
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        user_agent=args.user_agent,
    )

    # Override with browser-like headers to reduce bot detection
    fetcher.session.headers.update(BROWSER_HEADERS)
    fetcher.user_agent = BROWSER_HEADERS["User-Agent"]

    # Warm cookies by visiting the homepage first
    print("🌐 Warming up: fetching Douban homepage for cookies...")
    try:
        fetcher.session.get(
            DOUBAN_HOMEPAGE,
            timeout=args.timeout,
            headers={"Referer": "https://www.douban.com"},
        )
        print("  ✓ Got initial cookies")
    except Exception as exc:
        print(f"  ⚠️  Homepage fetch failed: {exc}")

    crawler = CelebrityCrawler(
        fetcher=fetcher,
        celebrity_limit=args.celebrity_limit,
        movie_limit=args.movie_limit,
        checkpoint_path=checkpoint_path,
    )

    try:
        movies, top_celebrities = crawler.crawl()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — checkpoint saved. Resume with:")
        print(f"  python data_tools/celebrity_crawler.py")
        return 1

    report = write_celebrity_outputs(output_dir, movies, top_celebrities)

    print()
    print("=" * 50)
    print("✅ 豆瓣影人数据采集完成")
    print(f"  影人 (Top {args.celebrity_limit}): {report['celebrity_count']} 位")
    print(f"  电影: {report['movie_count']} 部")
    print(f"  图谱 CSV 行数: {report['kg_row_count']}")
    print(f"  输出目录: {output_dir.resolve()}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
