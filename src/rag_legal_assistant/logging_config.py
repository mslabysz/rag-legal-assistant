import logging

FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=FORMAT, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)