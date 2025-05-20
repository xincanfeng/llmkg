import json
import os
from logging import getLogger
import random


logger = getLogger(__name__)


class ArticleReader:
    def __init__(self, base_dir='data/sources', max_articles=1):
        self.max_articles = max_articles
        self.base_dir = base_dir

    def load_json_file(self, read_articles_file):
        """Load and parse the first 'max_articles' from a JSON file."""
        articles = []
        articles_path = os.path.join(self.base_dir, read_articles_file)
        try:
            with open(articles_path, 'r', encoding='utf-8') as file:
                # Load only the first 'max_articles' entries from JSON file
                data = json.load(file)
                # articles = data[:self.max_articles]  # Assume the JSON is an array of objects
                articles = random.sample(data, k=self.max_articles) if len(data) >= self.max_articles else data  
        except json.JSONDecodeError as e:
            logger.error(f"[STEP 2] Error decoding JSON from the file: {e}")
        except FileNotFoundError:
            logger.error(f"[STEP 2] File not found: {self.articles_path}")
        except Exception as e:
            logger.error(f"[STEP 2] An error occurred: {e}")
        return articles

    def display_articles(self, articles):
        """Print out the details of each article in a readable format."""
        for i, article in enumerate(articles):
            article_details = f"[STEP 2] Random sample article {i + 1}:\n"
            for key, value in article.items():
                article_details += f"  {key}: {value}\n"
            logger.info(article_details)

    def __call__(self, read_articles_file):
        if read_articles_file == 'none':
            logger.info("[STEP 2] Fetching articles, not reading")
        else:
            articles = self.load_json_file(read_articles_file)
            self.display_articles(articles)


if __name__ == "__main__":
    
    reader = ArticleReader()
    reader(read_articles_file='pubmed.json')  