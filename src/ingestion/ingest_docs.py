from src.ingestion.pipeline import IngestionPipeline
from src.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

URLS = [
    # Databricks — Delta Lake
    "https://docs.databricks.com/en/delta/index.html",
    "https://docs.databricks.com/en/delta/delta-intro.html",
    "https://docs.databricks.com/en/structured-streaming/index.html",
    # Apache Spark
    "https://spark.apache.org/docs/latest/sql-programming-guide.html",
    "https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html",
    # dbt
    "https://docs.getdbt.com/docs/introduction",
    "https://docs.getdbt.com/docs/build/models",
    "https://docs.getdbt.com/docs/build/tests",
    "https://docs.getdbt.com/docs/build/sources",
]


def main() -> None:
    pipeline = IngestionPipeline()
    total = 0

    for url in URLS:
        logger.info("ingesting", url=url)
        try:
            stored = pipeline.ingest_url(url)
            total += stored
            logger.info("ingested", url=url, chunks=stored)
        except Exception as e:
            logger.error("ingestion_failed", url=url, error=str(e))
            continue

    logger.info("ingestion_complete", total_chunks=total)
    print(f"\n✅ Ingestion terminée — {total} chunks stockés dans pgvector")


if __name__ == "__main__":
    main()
