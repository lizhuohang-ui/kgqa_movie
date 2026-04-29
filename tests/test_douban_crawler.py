import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tools.douban_crawler import (
    MovieRecord,
    build_kg_rows,
    extract_douban_id,
    is_disallowed_douban_path,
    parse_movie_detail,
    parse_top250_movie_links,
)


TOP250_HTML = """
<html>
  <body>
    <ol class="grid_view">
      <li>
        <a href="https://movie.douban.com/subject/1291546/" title="霸王别姬"></a>
        <p class="">1993&nbsp;/&nbsp;中国大陆 中国香港&nbsp;/&nbsp;剧情 爱情 同性</p>
        <span class="rating_num">9.6</span>
      </li>
      <li>
        <a href="https://movie.douban.com/subject/1292052/" title="肖申克的救赎"></a>
        <p class="">1994&nbsp;/&nbsp;美国&nbsp;/&nbsp;犯罪 剧情</p>
        <span class="rating_num">9.7</span>
      </li>
    </ol>
  </body>
</html>
"""


MOVIE_DETAIL_HTML = """
<html>
  <head>
    <title>霸王别姬 (豆瓣)</title>
    <script type="application/ld+json">
      {
        "@context": "http://schema.org",
        "@type": "Movie",
        "name": "霸王别姬",
        "url": "https://movie.douban.com/subject/1291546/",
        "datePublished": "1993-01-01",
        "genre": ["剧情", "爱情"],
        "description": "段小楼与程蝶衣半个世纪的悲欢离合。",
        "director": [
          {"@type": "Person", "name": "陈凯歌", "url": "https://movie.douban.com/celebrity/1023040/"}
        ],
        "actor": [
          {"@type": "Person", "name": "张国荣", "url": "https://movie.douban.com/celebrity/1003494/"},
          {"@type": "Person", "name": "巩俐", "url": "https://movie.douban.com/celebrity/1035641/"}
        ],
        "aggregateRating": {"ratingValue": "9.6", "ratingCount": "2420617"}
      }
    </script>
  </head>
  <body>
    <div id="info">
      制片国家/地区: 中国大陆 / 中国香港<br>
      片长: 171分钟
    </div>
  </body>
</html>
"""


class DoubanCrawlerTests(unittest.TestCase):
    def test_extracts_subject_links_from_top250_page(self):
        links = parse_top250_movie_links(TOP250_HTML)

        self.assertEqual(
            links,
            [
                "https://movie.douban.com/subject/1291546/",
                "https://movie.douban.com/subject/1292052/",
            ],
        )

    def test_parses_movie_detail_from_json_ld_and_info_text(self):
        movie = parse_movie_detail(MOVIE_DETAIL_HTML, "https://movie.douban.com/subject/1291546/")

        self.assertEqual(movie.title, "霸王别姬")
        self.assertEqual(movie.douban_id, "1291546")
        self.assertEqual(movie.year, "1993")
        self.assertEqual(movie.rating, "9.6")
        self.assertEqual(movie.directors, ["陈凯歌"])
        self.assertEqual(movie.actors, ["张国荣", "巩俐"])
        self.assertEqual(movie.genres, ["剧情", "爱情"])
        self.assertEqual(movie.countries, ["中国大陆", "中国香港"])
        self.assertEqual(
            movie.celebrity_urls,
            [
                "https://movie.douban.com/celebrity/1003494/",
                "https://movie.douban.com/celebrity/1035641/",
            ],
        )

    def test_builds_current_graph_compatible_rows(self):
        movie = MovieRecord(
            title="霸王别姬",
            douban_id="1291546",
            url="https://movie.douban.com/subject/1291546/",
            directors=["陈凯歌"],
            actors=["张国荣", "巩俐"],
            genres=["剧情", "爱情"],
            countries=["中国大陆", "中国香港"],
            year="1993",
            rating="9.6",
        )

        rows = build_kg_rows([movie])

        self.assertEqual(
            rows,
            [
                {
                    "title": "霸王别姬",
                    "director": "陈凯歌",
                    "actor": "张国荣",
                    "genre": "剧情",
                    "year": "1993",
                    "rating": "9.6",
                    "country": "中国大陆",
                    "douban_id": "1291546",
                    "douban_url": "https://movie.douban.com/subject/1291546/",
                },
                {
                    "title": "霸王别姬",
                    "director": "陈凯歌",
                    "actor": "巩俐",
                    "genre": "剧情",
                    "year": "1993",
                    "rating": "9.6",
                    "country": "中国大陆",
                    "douban_id": "1291546",
                    "douban_url": "https://movie.douban.com/subject/1291546/",
                },
            ],
        )

    def test_blocks_douban_search_and_json_api_paths(self):
        self.assertTrue(is_disallowed_douban_path("https://movie.douban.com/subject_search?search_text=张国荣"))
        self.assertTrue(is_disallowed_douban_path("https://movie.douban.com/j/search_subjects"))
        self.assertTrue(is_disallowed_douban_path("https://www.douban.com/search?q=电影"))
        self.assertFalse(is_disallowed_douban_path("https://movie.douban.com/top250"))

    def test_extracts_douban_ids(self):
        self.assertEqual(extract_douban_id("https://movie.douban.com/subject/1291546/"), "1291546")
        self.assertEqual(extract_douban_id("https://movie.douban.com/celebrity/1003494/"), "1003494")


if __name__ == "__main__":
    unittest.main()
