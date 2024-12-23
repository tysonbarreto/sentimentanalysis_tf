import logging
from logging import StreamHandler, FileHandler
import os
from datetime import datetime
import sys
from pathlib import Path

def logger():
    LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y')}.log"

    LOG_PATH=os.path.join(os.getcwd(),"logs",LOG_FILE)
    os.makedirs(os.path.dirname(LOG_PATH),exist_ok=True)


    stream_handler = StreamHandler(sys.stdout)
    file_handler = FileHandler(LOG_PATH)

    logging.basicConfig(
        #filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            stream_handler,
            file_handler
        ]
    )

    return logging.getLogger('ml_classifier')

if __name__=="__main__":
    __all__=["logger"]