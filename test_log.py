import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("Test")

print("Print statement")
logger.info("Log statement")
sys.stdout.flush()
sys.stderr.flush()
