import asyncio
from logging import getLogger
import math
import os
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from Bio import Entrez, Medline
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm

# Constants
DEFAULT_MAX_RETRY = 50
DEFAULT_RETRY_BACKOFF_FACTOR = 0.3
DEFAULT_RETRY_STATUS_FORCELIST = [500, 502, 503, 504]
DEFAULT_MAX_DAILY_REQUESTS = 10000
DEFAULT_REQUEST_DELAY = 0.15
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_WORKERS = 4
SEARCH_MULTIPLIER = 1.5


# If get all logs, set logging level to logging.DEBUG
logger = getLogger(__name__)


class EntrezFetcher:
    """A class to fetch articles from Entrez databases using the BioPython Entrez module.

    This class handles searching for articles, fetching abstracts, and saving the results.
    It includes rate limiting and progress tracking features.

    Attributes:
        articles_file (str): Path to save the fetched articles.
        dataset (str): Name of the dataset.
        save_dir (str): Directory to save the fetched articles.
        max_retry (int): Maximum number of retries for failed requests.
        retry_backoff_factor (float): Backoff factor for retry mechanism.
        retry_status_forcelist (List[int]): List of HTTP status codes to force a retry.
        max_daily_requests (int): Maximum number of requests per day.
        request_delay (float): Delay between requests to avoid rate limiting.
        timeout (float): Timeout for requests.
        max_workers (int): Maximum number of workers for concurrent requests.
    """

    def __init__(
        self,
        articles_file: str,
        current_time: str,
        keywords_file_without_extension: str,
        log_file: str, 
        dataset: str,
        save_dir: str,
        email: str,
        api_key: str,
        max_retry: int = DEFAULT_MAX_RETRY,
        retry_backoff_factor: float = DEFAULT_RETRY_BACKOFF_FACTOR,
        retry_status_forcelist: List[int] = DEFAULT_RETRY_STATUS_FORCELIST,
        max_daily_requests: int = DEFAULT_MAX_DAILY_REQUESTS,
        request_delay: float = DEFAULT_REQUEST_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
        max_workers: int = DEFAULT_MAX_WORKERS,
        ) -> None:
        """Initialize the EntrezFetcher object.

        Args:
            articles_file (str): Path to save the fetched articles.
            current_time (str): Current time.
            keywords_file_without_extension (str): Name of the keywords file without extension.
            log_file (str): Name of the log file.
            dataset (str): Name of the dataset.
            save_dir (str): Directory to save the fetched articles.
            email (str): Email address for Entrez API.
            api_key (str): API key for Entrez API.
            max_retry (int, optional): Maximum number of retries. Defaults to DEFAULT_MAX_RETRY.
            retry_backoff_factor (float, optional): Backoff factor for retry. Defaults to DEFAULT_RETRY_BACKOFF_FACTOR.
            retry_status_forcelist (List[int], optional): HTTP status codes to force retry. Defaults to DEFAULT_RETRY_STATUS_FORCELIST.
            max_daily_requests (int, optional): Maximum daily requests. Defaults to DEFAULT_MAX_DAILY_REQUESTS.
            request_delay (float, optional): Delay between requests. Defaults to DEFAULT_REQUEST_DELAY.
            timeout (float, optional): Request timeout. Defaults to DEFAULT_TIMEOUT.
            max_workers (int, optional): Maximum number of workers. Defaults to DEFAULT_MAX_WORKERS.
        """
        # Set up Entrez email and API key
        Entrez.email = email
        Entrez.api_key = api_key
        
        self.log_dir = os.path.join(save_dir, dataset, f'{current_time}_{keywords_file_without_extension}_{log_file}')
        self.articles_path = os.path.join(self.log_dir, articles_file)
        self.available_dbs = self._get_entrez_databases()
        self.max_retry = max_retry
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_status_forcelist = retry_status_forcelist
        self.max_daily_requests = max_daily_requests
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_workers = max_workers
        self.last_request_time = 0
        self.daily_request_count = 0
        self.last_request_date: Optional[str] = None
        self.failed_article_ids: Set[str] = set()

        # Set up requests session with retry mechanism
        self.session = self._setup_session()

        # New attributes for improved rate limiting
        self.request_count = 0
        self.request_lock = threading.Lock()
        self.request_semaphore = asyncio.Semaphore(max_workers)
        self.last_reset_time = time.time()

        logger.info(f"Using Entrez email: {Entrez.email}")

    def _setup_session(self) -> requests.Session:
        """Set up a requests session with retry mechanism.

        Returns:
            requests.Session: Configured session object.
        """
        session = requests.Session()
        retry = Retry(
            total=self.max_retry,
            backoff_factor=self.retry_backoff_factor,
            status_forcelist=self.retry_status_forcelist,
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    async def _rate_limit(self) -> None:
        """Apply rate limiting to avoid exceeding the API request limits."""
        async with self.request_semaphore:
            current_time = time.time()
            requests_per_second = 10 if Entrez.email != "A.N.Other@example.com" else 3

            with self.request_lock:
                if current_time - self.last_reset_time >= 1:
                    self.request_count = 0
                    self.last_reset_time = current_time

                if self.request_count >= requests_per_second:
                    wait_time = 1 - (current_time - self.last_reset_time)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    self.request_count = 0
                    self.last_reset_time = time.time()

                self.request_count += 1

        # Check daily limit
        current_date = time.strftime("%Y-%m-%d")
        self._reset_daily_count_if_new_day(current_date)
        self._check_daily_limit(current_time, current_date)

    def _reset_daily_count_if_new_day(self, current_date: str) -> None:
        """Reset the daily request count if a new day has started.

        Args:
            current_date (str): The current date.
        """
        if self.last_request_date != current_date:
            self.daily_request_count = 0
            self.last_request_date = current_date

    def _check_daily_limit(self, current_time: float, current_date: str) -> None:
        """Check if the daily request limit has been reached.

        Args:
            current_time (float): The current time.
            current_date (str): The current date.
        """
        if self.daily_request_count >= self.max_daily_requests:
            wait_time = self._calculate_wait_time(current_time, current_date)
            logger.warning(f"[STEP 2] Daily request limit reached. Waiting for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
            self.daily_request_count = 0
            self.last_request_date = time.strftime("%Y-%m-%d")

    def _calculate_wait_time(self, current_time: float, current_date: str) -> float:
        """Calculate the wait time until the next day.

        Args:
            current_time (float): The current time.
            current_date (str): The current date.

        Returns:
            float: The wait time in seconds.
        """
        return 86400 - (current_time - time.mktime(time.strptime(current_date, "%Y-%m-%d")))

    @staticmethod
    def _get_entrez_databases() -> List[str]:
        """Get a list of available Entrez databases.

        Returns:
            List[str]: List of available Entrez database names.
        """
        handle = Entrez.einfo()
        result = handle.read()
        root = ET.fromstring(result)
        databases = [db.text for db in root.findall(".//DbName")]
        logger.info(f"Available Entrez databases: {', '.join(databases)}")
        return databases

    async def search_articles(self, keywords: List[str], db: str, retmax: int) -> List[str]:
        """Search for articles using the specified keywords in the specified Entrez database.

        Args:
            keywords (List[str]): List of keywords to search for.
            db (str): Entrez database to search in.
            retmax (int): Maximum number of articles to fetch.

        Returns:
            List[str]: List of article IDs found.
        """
        await self._rate_limit()
        query = " AND ".join(keywords)
        adjusted_retmax = math.ceil(retmax * SEARCH_MULTIPLIER)
        try:
            logger.info(f"[STEP 2] Searching for articles with keywords: {query} in {db}")
            handle = await asyncio.to_thread(Entrez.esearch, db=db, term=query, retmax=adjusted_retmax)
            record = await asyncio.to_thread(Entrez.read, handle)
            await asyncio.to_thread(handle.close)
            logger.info(f"[STEP 2] Found {len(record['IdList'])} articles.")
            return record["IdList"]
        except Exception as e:
            logger.error(f"[STEP 2] Error in search_articles: {e}")
            logger.error(f"[STEP 2] Request details: DB={db}, Keywords={query}, RetMax={adjusted_retmax}")
            return []

    async def fetch_abstract(self, article_id: str, db: str) -> Dict[str, str]:
        """Fetch abstract for a given article ID.

        Args:
            article_id (str): Article ID to fetch.
            db (str): Entrez database to search in.

        Returns:
            Dict[str, str]: Fetched article information.
        """
        await self._rate_limit()
        try:
            logger.debug(f"[STEP 2] Fetching abstract for article ID: {article_id}")
            handle = await asyncio.to_thread(Entrez.efetch, db=db, id=article_id, rettype="medline", retmode="text", timeout=self.timeout)
            records = await asyncio.to_thread(list, Medline.parse(handle))
            await asyncio.to_thread(handle.close)

            if records:
                return self._process_record(records[0], article_id)
            else:
                self.failed_article_ids.add(article_id)
                return {}

        except requests.exceptions.Timeout:
            self.failed_article_ids.add(article_id)
            return {}
        except Exception as e:
            self.failed_article_ids.add(article_id)
            if "HTTP Error 400" in str(e):
                logger.debug(f"[STEP 2] Invalid request for article ID: {article_id}. Skipping...")
            return {}

    @staticmethod
    def _process_record(record: Dict[str, Any], article_id: str) -> Dict[str, str]:
        """Process a fetched article record.

        Args:
            record (Dict[str, Any]): Fetched article record.
            article_id (str): Article ID.

        Returns:
            Dict[str, str]: Processed article information.
        """
        keywords = record.get("OT", [])
        keywords_str = ", ".join(keywords)
        return {
            "ArticleID": article_id,
            "Title": record.get("TI", ""),
            "Keywords": keywords_str,
            "Abstract": record.get("AB", ""),
            }

    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save the fetched articles to a JSON file.

        Args:
            results_df (pd.DataFrame): DataFrame containing the fetched articles.
            articles_path (str): Path to the JSON file to save the articles.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        results_df.to_json(self.articles_path, orient="records", indent=4)

    async def process_database(self, keywords: List[str], db: str, retmax: int, overall_progress: tqdm) -> List[Dict[str, str]]:
        """Process a single database to fetch articles.

        Args:
            keywords (List[str]): List of keywords to search for.
            db (str): Entrez database to search in.
            retmax (int): Maximum number of articles to fetch.
            overall_progress (tqdm): Overall progress bar.

        Returns:
            List[Dict[str, str]]: List of fetched and processed articles.
        """
        article_ids = await self.search_articles(keywords, db, retmax)

        if not article_ids:
            logger.warning(f"[STEP 2] No articles found for keywords: {keywords} in {db}.")
            return []

        articles = []
        fetch_progress = tqdm(total=len(article_ids), desc=f"Fetching from {db}", position=2, leave=False)

        for article_id in article_ids:
            article = await self.fetch_abstract(article_id, db)
            if article:
                articles.append(article)
                overall_progress.update(1)
                fetch_progress.update(1)

        fetch_progress.close()
        return self._process_articles(articles, db, keywords)[:retmax]

    def _process_articles(self, articles: List[Dict[str, str]], db: str, keywords: List[str]) -> List[Dict[str, str]]:
        """Process fetched articles by adding metadata.

        Args:
            articles (List[Dict[str, str]]): List of fetched articles.
            db (str): Entrez database used.
            keywords (List[str]): Search keywords used.

        Returns:
            List[Dict[str, str]]: Processed articles with added metadata.
        """
        for article in articles:
            article["Database"] = db
            article["SearchKeywords"] = ", ".join(keywords)
            article["KeywordsMatch"] = any(
                keyword.lower() in article["Keywords"].lower()
                for keyword in keywords
                )
        return articles

    async def async_call(self, keywords: List[str], dbs: List[str] = ["pubmed"], retmax: int = 50) -> pd.DataFrame:
        """Asynchronously fetch articles from Entrez databases.

        Args:
            keywords (List[str]): List of keywords to search for.
            dbs (List[str], optional): List of Entrez databases to search in. Defaults to ["pubmed"].
            retmax (int, optional): Maximum number of articles to fetch per database. Defaults to 50.

        Returns:
            pd.DataFrame: DataFrame containing the fetched articles.
        """
        all_data = []
        total_articles = len(dbs) * retmax

        overall_progress = tqdm(total=total_articles, desc="Overall Progress", position=0)
        db_progress = tqdm(total=len(dbs), desc="Processing Databases", position=1, leave=False)

        for db in dbs:
            db_data = await self.process_database(keywords, db, retmax, overall_progress)
            all_data.extend(db_data)
            db_progress.update(1)

        overall_progress.close()
        db_progress.close()

        df = pd.DataFrame(all_data)
        self.save_results(results_df=df)

        if self.failed_article_ids:
            logger.info(f"[STEP 2] Failed to fetch {len(self.failed_article_ids)} articles:")
            for failed_id in self.failed_article_ids:
                logger.info(f"[STEP 2]  - {failed_id}")

        return df

    def __call__(self, keywords: List[str], dbs: List[str] = ["pubmed"], retmax: int = 50) -> pd.DataFrame:
        """Fetch articles from Entrez databases using the specified keywords.

        This method is the main entry point for the EntrezFetcher class. It handles the asynchronous
        execution of the article fetching process.

        Args:
            keywords (list): List of keywords to search for in PubMed.
            
            dbs (list): List of databases to fetch abstracts from.
            dbs (list): choose from 
            ["pubmed", "protein", "nuccore", "ipg", "nucleotide", "structure", "genome", "annotinfo", 
            "assembly", "bioproject", "biosample", "blastdbinfo", "books", "cdd", "clinvar", "gap", 
            "gapplus", "grasp", "dbvar", "gene", "gds", "geoprofiles", "medgen", "mesh", "nlmcatalog", 
            "omim", "orgtrack", "pmc", "popset", "proteinclusters", "pcassay", "protfam", "pccompound", 
            "pcsubstance", "seqannot", "snp", "sra", "taxonomy", "biocollections", "gtr"],

            retmax (int): Maximum number of abstracts to retrieve.

        Returns:
            pd.DataFrame: DataFrame containing fetched abstracts. Following columns are added:
                - ArticleID: Article ID.
                - Title: Article title.
                - Keywords: Article keywords.
                - Abstract: Article abstract.
                - Database: Database used to fetch the article.
                - SearchKeywords: Search keywords used to fetch the article.
                - KeywordsMatch: Flag indicating whether the search keywords match the article keywords.
        """
        print(f"Available Databases: \n{self.available_dbs}")
        # logger.info(f"[STEP 2] Using Entrez databases: {dbs}")
        return asyncio.run(self.async_call(keywords, dbs, retmax))


if __name__ == "__main__":
    fetcher = EntrezFetcher(
        articles_file="articles.json",
        current_time='20241006', 
        keywords_file_without_extension='h_r_t', 
        log_file='log', 
        dataset="UMLS",
        save_dir="./output",
        email="",
        api_key="609c1ed209c75a029954dc1b546e065c2408",
        )
    df = fetcher(keywords=["protein", "structure"], dbs=["pubmed"], retmax=25)