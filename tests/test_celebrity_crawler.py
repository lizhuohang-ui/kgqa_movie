import json
import tempfile
import unittest
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tools.celebrity_crawler import (
    CelebrityRef,
    _init_state,
    _load_checkpoint,
    _save_checkpoint,
    is_douban_challenge_page,
    parse_celebrity_filmography_page,
    write_celebrity_outputs,
)
from data_tools.douban_crawler import (
    KG_CSV_FIELDS,
    MovieRecord,
    build_kg_rows,
)


FILMOGRAPHY_HTML = """
<html>
<body>
  <div id="wrapper">
    <div class="article">
      <ul class="list-view">
        <li class="item">
          <a href="https://movie.douban.com/subject/1291546/?from=celebrity"
             title="霸王别姬">霸王别姬</a>
          <span>1993</span>
        </li>
        <li class="item">
          <a href="https://movie.douban.com/subject/1292052/?from=celebrity"
             title="肖申克的救赎">肖申克的救赎</a>
          <span>1994</span>
        </li>
        <li class="item">
          <a href="https://movie.douban.com/subject/1291546/?from=celebrity"
             title="霸王别姬（重复）">霸王别姬</a>
          <span>1993</span>
        </li>
      </ul>
    </div>
  </div>
</body>
</html>
"""

CHALLENGE_HTML = """
<!DOCTYPE html>
<html>
<head><title>豆瓣</title></head>
<body>
  <div>
    <p class="loading">载入中 ...</p>
    <form name="sec" id="sec" method="POST" action="/c">
      <input type="hidden" id="tok" name="tok" value="test-token" />
    </form>
  </div>
</body>
</html>
"""

NORMAL_HTML = """
<!DOCTYPE html>
<html>
<head><title>电影详情</title></head>
<body>
  <h1>霸王别姬</h1>
  <script type="application/ld+json">
  {
    "@context": "http://schema.org",
    "@type": "Movie",
    "name": "霸王别姬",
    "url": "https://movie.douban.com/subject/1291546/",
    "director": [
      {"@type": "Person", "name": "陈凯歌", "url": "https://movie.douban.com/celebrity/1023040/"}
    ],
    "actor": [
      {"@type": "Person", "name": "张国荣", "url": "https://movie.douban.com/celebrity/1003494/"},
      {"@type": "Person", "name": "巩俐", "url": "https://movie.douban.com/celebrity/1035641/"}
    ],
    "aggregateRating": {"ratingValue": "9.6"},
    "datePublished": "1993-01-01",
    "genre": ["剧情", "爱情"]
  }
  </script>
</body>
</html>
"""


class ParseCelebrityFilmographyPageTests(unittest.TestCase):
    def test_extracts_unique_movie_links_by_douban_id(self):
        refs = parse_celebrity_filmography_page(
            FILMOGRAPHY_HTML,
            base_url="https://movie.douban.com/celebrity/1003494/movies",
        )

        ids = [r["douban_id"] for r in refs]
        self.assertEqual(ids, ["1291546", "1292052"])

    def test_strips_query_strings_from_urls(self):
        refs = parse_celebrity_filmography_page(
            FILMOGRAPHY_HTML,
            base_url="https://movie.douban.com/celebrity/1003494/movies",
        )

        for ref in refs:
            self.assertNotIn("?from=celebrity", ref["url"])
            self.assertEqual(ref["url"], f"https://movie.douban.com/subject/{ref['douban_id']}/")

    def test_returns_empty_list_for_no_subject_links(self):
        html = "<html><body><a href='/celebrity/123/'>Actor</a></body></html>"
        refs = parse_celebrity_filmography_page(
            html,
            base_url="https://movie.douban.com/celebrity/1003494/movies",
        )
        self.assertEqual(refs, [])


class IsDoubanChallengePageTests(unittest.TestCase):
    def test_detects_challenge_page_with_form_name_sec(self):
        self.assertTrue(is_douban_challenge_page(CHALLENGE_HTML))

    def test_detects_chinese_warning_text(self):
        html = '<html>检测到有异常请求从你的 IP 发出，请验证你为正常人</html>'
        self.assertTrue(is_douban_challenge_page(html))

    def test_returns_false_for_normal_page(self):
        self.assertFalse(is_douban_challenge_page(NORMAL_HTML))

    def test_returns_false_for_empty_html(self):
        self.assertFalse(is_douban_challenge_page(""))
        self.assertFalse(is_douban_challenge_page(None))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.tmpdir.name) / "checkpoint.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_state_has_expected_keys(self):
        state = _init_state()
        for key in (
            "version", "phase", "top_celebrities", "celebrity_index",
            "discovered_movie_urls", "movie_index", "movies", "stats",
        ):
            self.assertIn(key, state)
        self.assertEqual(state["phase"], 1)

    def test_save_and_load_roundtrip(self):
        state = _init_state()
        state["phase"] = 2
        state["top_celebrities"] = [
            CelebrityRef(name="陈凯歌", douban_id="1023040",
                         url="https://movie.douban.com/celebrity/1023040/",
                         top250_count=3),
        ]
        state["discovered_movie_urls"] = [
            {"douban_id": "1291546",
             "url": "https://movie.douban.com/subject/1291546/",
             "title": "霸王别姬"},
        ]
        state["movies"] = [
            MovieRecord(
                title="霸王别姬", douban_id="1291546",
                url="https://movie.douban.com/subject/1291546/",
                directors=["陈凯歌"], actors=["张国荣", "巩俐"],
                genres=["剧情", "爱情"], countries=["中国大陆"],
                year="1993", rating="9.6",
            ),
        ]
        state["stats"]["top250_movies_crawled"] = 5

        _save_checkpoint(state, self.checkpoint_path)
        loaded = _load_checkpoint(self.checkpoint_path)

        self.assertEqual(loaded["phase"], 2)
        self.assertEqual(len(loaded["top_celebrities"]), 1)
        self.assertIsInstance(loaded["top_celebrities"][0], CelebrityRef)
        self.assertEqual(loaded["top_celebrities"][0].name, "陈凯歌")
        self.assertEqual(len(loaded["movies"]), 1)
        self.assertIsInstance(loaded["movies"][0], MovieRecord)
        self.assertEqual(loaded["movies"][0].title, "霸王别姬")
        self.assertEqual(loaded["stats"]["top250_movies_crawled"], 5)

    def test_load_returns_none_for_missing_file(self):
        self.assertIsNone(_load_checkpoint(Path("/nonexistent/checkpoint.json")))

    def test_load_returns_none_for_corrupt_file(self):
        self.checkpoint_path.write_text("{not valid json", encoding="utf-8")
        self.assertIsNone(_load_checkpoint(self.checkpoint_path))


class WriteCelebrityOutputsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)

        self.movies = [
            MovieRecord(
                title="霸王别姬", douban_id="1291546",
                url="https://movie.douban.com/subject/1291546/",
                directors=["陈凯歌"], actors=["张国荣", "巩俐"],
                genres=["剧情", "爱情"], countries=["中国大陆"],
                year="1993", rating="9.6",
            ),
        ]
        self.celebrities = [
            CelebrityRef(
                name="陈凯歌", douban_id="1023040",
                url="https://movie.douban.com/celebrity/1023040/",
                top250_count=3,
            ),
        ]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_writes_all_output_files(self):
        report = write_celebrity_outputs(
            self.output_dir, self.movies, self.celebrities
        )

        self.assertTrue((self.output_dir / "movies.json").exists())
        self.assertTrue((self.output_dir / "celebrities.json").exists())
        self.assertTrue((self.output_dir / "movies_data_douban.csv").exists())
        self.assertTrue((self.output_dir / "crawl_report.json").exists())

        self.assertEqual(report["movie_count"], 1)
        self.assertEqual(report["celebrity_count"], 1)

    def test_csv_has_correct_columns(self):
        write_celebrity_outputs(self.output_dir, self.movies, self.celebrities)
        csv_path = self.output_dir / "movies_data_douban.csv"

        content = csv_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        header = lines[0].split(",")

        for field in KG_CSV_FIELDS:
            self.assertIn(field, header)

    def test_csv_rows_match_build_kg_rows(self):
        write_celebrity_outputs(self.output_dir, self.movies, self.celebrities)
        csv_path = self.output_dir / "movies_data_douban.csv"

        content = csv_path.read_text(encoding="utf-8")
        data_lines = [line for line in content.strip().split("\n")[1:] if line.strip()]

        expected_rows = build_kg_rows(self.movies)
        self.assertEqual(len(data_lines), len(expected_rows))


class CelebrityRefTests(unittest.TestCase):
    def test_asdict_serialization(self):
        ref = CelebrityRef(
            name="陈凯歌", douban_id="1023040",
            url="https://movie.douban.com/celebrity/1023040/",
            top250_count=3,
        )
        d = asdict(ref)
        self.assertEqual(d["name"], "陈凯歌")
        self.assertEqual(d["douban_id"], "1023040")
        self.assertEqual(d["top250_count"], 3)

    def test_roundtrip_via_json(self):
        ref = CelebrityRef(
            name="张国荣", douban_id="1003494",
            url="https://movie.douban.com/celebrity/1003494/",
            top250_count=4,
        )
        serialized = json.dumps(asdict(ref), ensure_ascii=False)
        d = json.loads(serialized)
        restored = CelebrityRef(**d)
        self.assertEqual(restored.name, ref.name)
        self.assertEqual(restored.douban_id, ref.douban_id)
        self.assertEqual(restored.top250_count, ref.top250_count)


if __name__ == "__main__":
    unittest.main()
