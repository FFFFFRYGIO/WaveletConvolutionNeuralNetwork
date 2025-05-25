"""Logger object for logging data operation, network creation and learning processes."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='[%(asctime)s] %(levelname)s %(funcName)s: %(message)s',
    datefmt='%y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
