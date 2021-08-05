import time
from bigmodels.models.cpm2 import CPM2, CPM2Configuration
import logging
logging.basicConfig(level=logging.INFO)

def main():
    config = CPM2Configuration()
    cpm2 = CPM2(config)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()