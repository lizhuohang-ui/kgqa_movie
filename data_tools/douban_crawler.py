import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests


DEFAULT_OUTPUT_DIR = "data/douban"
DEFAULT_DELAY_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_USER_AGENT = "kgqa-movie-research-crawler/1.0"
TOP250_URL = "https://movie.douban.com/top250"

DISALLOWED_DOUBAN_PATHS = (
    "/subject_search",
    "/search",
    "/celebrities/search",
    "/j/",
)

KG_CSV_FIELDS = [
    "title",
    "director",
    "actor",
    "genre",
    "year",
    "rating",
    "country",
    "douban_id",
    "douban_url",
]


@dataclass
class MovieRecord:
    title: str
    douban_id: str
    url: str
    directors: list[str] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    year: str = ""
    rating: str = ""
    rating_count: str = ""
    summary: str = ""
    celebrity_urls: list[str] = field(default_factory=list)


@dataclass
class ActorRecord:
    name: str
    douban_id: str
    url: str
    gender: str = ""
    birthplace: str = ""
    birth_date: str = ""
    professions: list[str] = field(default_factory=list)
    related_movies: list[str] = field(default_factory=list)


def normalize_space(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def unique_keep_order(values):
    seen = set()
    result = []
    for value in values:
        cleaned = normalize_space(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def extract_douban_id(url):
    match = re.search(r"/(?:subject|celebrity)/(\d+)/?", url)
    return match.group(1) if match else ""


def is_disallowed_douban_path(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host.endswith("douban.com"):
        return False

    path = parsed.path
    return any(path.startswith(disallowed) for disallowed in DISALLOWED_DOUBAN_PATHS)


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return

        attr_map = dict(attrs)
        href = attr_map.get("href", "")
        if not href:
            return

        self._current = {
            "href": href,
            "title": attr_map.get("title", ""),
            "text_parts": [],
        }

    def handle_data(self, data):
        if self._current:
            self._current["text_parts"].append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._current:
            text = normalize_space("".join(self._current["text_parts"]))
            self.links.append(
                {
                    "href": self._current["href"],
                    "title": normalize_space(self._current["title"] or text),
                    "text": text,
                }
            )
            self._current = None


class TextAndScriptParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.json_ld_scripts = []
        self._script_type = ""
        self._script_parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag_name = tag.lower()
        if tag_name in {"style", "noscript"}:
            self._skip_depth += 1
            return

        if tag_name == "script":
            self._script_type = dict(attrs).get("type", "").lower()
            self._script_parts = []

    def handle_data(self, data):
        if self._script_type:
            self._script_parts.append(data)
            return

        if not self._skip_depth:
            self.text_parts.append(data)

    def handle_endtag(self, tag):
        tag_name = tag.lower()
        if tag_name in {"style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return

        if tag_name == "script":
            if "application/ld+json" in self._script_type:
                self.json_ld_scripts.append("".join(self._script_parts))
            self._script_type = ""
            self._script_parts = []

    @property
    def text(self):
        return normalize_space(" ".join(self.text_parts))


def _parse_json_ld(html):
    parser = TextAndScriptParser()
    parser.feed(html)

    for script in parser.json_ld_scripts:
        cleaned = script.strip()
        if not cleaned:
            continue
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item
    return {}


def _collect_links(html, base_url):
    parser = LinkParser()
    parser.feed(html)
    links = []
    for link in parser.links:
        links.append(
            {
                "href": urljoin(base_url, link["href"]),
                "title": link["title"],
                "text": link["text"],
            }
        )
    return links


def parse_top250_movie_links(html, base_url=TOP250_URL):
    links = _collect_links(html, base_url)
    subject_urls = []
    for link in links:
        href = link["href"]
        if re.search(r"/subject/\d+/?", href):
            subject_urls.append(href.split("?")[0])
    return unique_keep_order(subject_urls)


def _people_from_json_ld(value):
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return [], []

    names = []
    urls = []
    for person in value:
        if not isinstance(person, dict):
            continue
        names.append(person.get("name", ""))
        url = person.get("url", "")
        if url:
            urls.append(url)
    return unique_keep_order(names), unique_keep_order(urls)


def _field_from_text(text, field_name):
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?:\s+[^\s:：]+[:：]|$)"
    match = re.search(pattern, text)
    return normalize_space(match.group(1)) if match else ""


def _split_slash_list(value):
    return unique_keep_order(re.split(r"\s*/\s*|\s+/\s+", value))


def _year_from_value(value):
    match = re.search(r"\d{4}", value or "")
    return match.group(0) if match else ""


def parse_movie_detail(html, url):
    data = _parse_json_ld(html)
    page_parser = TextAndScriptParser()
    page_parser.feed(html)
    text = page_parser.text

    director_names, _director_urls = _people_from_json_ld(data.get("director"))
    actor_names, actor_urls = _people_from_json_ld(data.get("actor"))

    rating_data = data.get("aggregateRating") if isinstance(data.get("aggregateRating"), dict) else {}
    rating = normalize_space(rating_data.get("ratingValue", ""))
    rating_count = normalize_space(rating_data.get("ratingCount", ""))

    genres = data.get("genre", [])
    if isinstance(genres, str):
        genres = _split_slash_list(genres)

    countries = _split_slash_list(_field_from_text(text, "制片国家/地区"))

    if not actor_urls:
        actor_urls = [
            link["href"]
            for link in _collect_links(html, url)
            if re.search(r"/celebrity/\d+/?", link["href"])
        ]

    return MovieRecord(
        title=normalize_space(data.get("name", "")),
        douban_id=extract_douban_id(url),
        url=url,
        directors=director_names,
        actors=actor_names,
        genres=unique_keep_order(genres),
        countries=countries,
        year=_year_from_value(data.get("datePublished", "")),
        rating=rating,
        rating_count=rating_count,
        summary=normalize_space(data.get("description", "")),
        celebrity_urls=unique_keep_order(actor_urls),
    )


def parse_actor_detail(html, url):
    parser = TextAndScriptParser()
    parser.feed(html)
    text = parser.text

    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = normalize_space(title_match.group(1) if title_match else "")
    title = re.sub(r"\s*\(豆瓣\)\s*$", "", title).strip()

    professions = _split_slash_list(_field_from_text(text, "职业"))
    movie_ids = [
        extract_douban_id(link["href"])
        for link in _collect_links(html, url)
        if re.search(r"/subject/\d+/?", link["href"])
    ]

    return ActorRecord(
        name=title,
        douban_id=extract_douban_id(url),
        url=url,
        gender=_field_from_text(text, "性别"),
        birthplace=_field_from_text(text, "出生地"),
        birth_date=_field_from_text(text, "出生日期"),
        professions=professions,
        related_movies=unique_keep_order(movie_ids),
    )


def is_chinese_movie(movie):
    return any("中国" in country for country in movie.countries)


def build_kg_rows(movies: Iterable[MovieRecord]):
    rows = []
    for movie in movies:
        directors = movie.directors or [""]
        actors = movie.actors or [""]
        genre = movie.genres[0] if movie.genres else ""
        country = movie.countries[0] if movie.countries else ""
        for director in directors:
            for actor in actors:
                rows.append(
                    {
                        "title": movie.title,
                        "director": director,
                        "actor": actor,
                        "genre": genre,
                        "year": movie.year,
                        "rating": movie.rating,
                        "country": country,
                        "douban_id": movie.douban_id,
                        "douban_url": movie.url,
                    }
                )
    return rows


class PoliteFetcher:
    def __init__(
        self,
        cache_dir,
        delay_seconds=DEFAULT_DELAY_SECONDS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        user_agent=DEFAULT_USER_AGENT,
        obey_robots=True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.obey_robots = obey_robots
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.last_request_at = {}
        self.robot_parsers = {}

    def fetch(self, url):
        if is_disallowed_douban_path(url):
            raise ValueError(f"Blocked by local Douban robots policy: {url}")
        if self.obey_robots and not self._can_fetch(url):
            raise ValueError(f"Blocked by robots.txt: {url}")

        cache_path = self._cache_path(url)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        self._sleep_for_host(url)
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        html = response.text
        cache_path.write_text(html, encoding="utf-8")
        return html

    def _cache_path(self, url):
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html"

    def _sleep_for_host(self, url):
        host = urlparse(url).netloc
        now = time.monotonic()
        last = self.last_request_at.get(host)
        if last is not None:
            wait_seconds = self.delay_seconds - (now - last)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        self.last_request_at[host] = time.monotonic()

    def _can_fetch(self, url):
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = self.robot_parsers.get(robots_url)
        if parser is None:
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
            except Exception:
                return False
            self.robot_parsers[robots_url] = parser
        return parser.can_fetch(self.user_agent, url)


class DoubanCrawler:
    def __init__(self, fetcher, movie_limit=100, actor_limit=100, chinese_only=True):
        self.fetcher = fetcher
        self.movie_limit = movie_limit
        self.actor_limit = actor_limit
        self.chinese_only = chinese_only

    def crawl(self, seed_urls=None):
        movie_urls = self._default_movie_urls()
        for url in seed_urls or []:
            if re.search(r"/subject/\d+/?", url):
                movie_urls.insert(0, url)

        movies = []
        actor_urls = []
        seen_movie_ids = set()

        for url in unique_keep_order(movie_urls):
            if len(movies) >= self.movie_limit:
                break
            html = self.fetcher.fetch(url)
            movie = parse_movie_detail(html, url)
            if not movie.title or movie.douban_id in seen_movie_ids:
                continue
            if self.chinese_only and not is_chinese_movie(movie):
                continue
            seen_movie_ids.add(movie.douban_id)
            movies.append(movie)
            actor_urls.extend(movie.celebrity_urls)

        actors = []
        seen_actor_ids = set()
        for url in unique_keep_order(actor_urls):
            if len(actors) >= self.actor_limit:
                break
            actor_id = extract_douban_id(url)
            if not actor_id or actor_id in seen_actor_ids:
                continue
            html = self.fetcher.fetch(url)
            actor = parse_actor_detail(html, url)
            if not actor.name:
                continue
            seen_actor_ids.add(actor_id)
            actors.append(actor)

        return movies, actors

    def _default_movie_urls(self):
        urls = []
        for start in range(0, 250, 25):
            list_url = f"{TOP250_URL}?start={start}&filter="
            html = self.fetcher.fetch(list_url)
            urls.extend(parse_top250_movie_links(html, list_url))
            if len(urls) >= self.movie_limit * 3:
                break
        return unique_keep_order(urls)


def read_seed_urls(seed_url_file):
    if not seed_url_file:
        return []
    path = Path(seed_url_file)
    if not path.exists():
        raise FileNotFoundError(seed_url_file)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def write_outputs(output_dir, movies, actors):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    movies_json = output_path / "movies.json"
    actors_json = output_path / "actors.json"
    kg_csv = output_path / "movies_data_douban.csv"
    report_json = output_path / "crawl_report.json"

    movies_json.write_text(
        json.dumps([asdict(movie) for movie in movies], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    actors_json.write_text(
        json.dumps([asdict(actor) for actor in actors], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = build_kg_rows(movies)
    with kg_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=KG_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "movie_count": len(movies),
        "actor_count": len(actors),
        "kg_row_count": len(rows),
        "files": {
            "movies": str(movies_json),
            "actors": str(actors_json),
            "kg_csv": str(kg_csv),
        },
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Crawl public Douban movie pages for KGQA expansion data.")
    parser.add_argument("--movie-limit", type=int, default=100, help="最多采集的中国电影数量")
    parser.add_argument("--actor-limit", type=int, default=100, help="最多采集的演员数量")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--cache-dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "cache"), help="HTML 缓存目录")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="同一域名请求间隔秒数")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="请求超时秒数")
    parser.add_argument("--seed-url-file", help="额外豆瓣 subject URL 文件，每行一个 URL")
    parser.add_argument("--include-non-chinese", action="store_true", help="不按国家/地区过滤中国电影")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="请求 User-Agent")
    return parser.parse_args()


def main():
    args = parse_args()
    fetcher = PoliteFetcher(
        cache_dir=args.cache_dir,
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        user_agent=args.user_agent,
    )
    crawler = DoubanCrawler(
        fetcher,
        movie_limit=args.movie_limit,
        actor_limit=args.actor_limit,
        chinese_only=not args.include_non_chinese,
    )
    movies, actors = crawler.crawl(seed_urls=read_seed_urls(args.seed_url_file))
    report = write_outputs(args.output_dir, movies, actors)

    print("豆瓣数据采集完成")
    print(f"电影: {report['movie_count']} 部")
    print(f"演员: {report['actor_count']} 位")
    print(f"图谱 CSV 行数: {report['kg_row_count']}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
