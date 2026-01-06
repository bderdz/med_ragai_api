import logging, os, sys

LOG_FILE = "logs/app.log"
METRICS_FILE = "logs/metrics.log"


def init_logging():
    # Logs configuration
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE)
        ]
    )

    logging.getLogger("src").setLevel(logging.DEBUG)
    # Separate logger for metrics
    metrics_logger = logging.getLogger("metrics")
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.propagate = False

    if not metrics_logger.handlers:
        metrics_handler = logging.FileHandler(METRICS_FILE)
        metrics_handler.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
        metrics_logger.addHandler(metrics_handler)
