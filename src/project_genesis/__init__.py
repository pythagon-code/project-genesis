import os
import logging
from logging.handlers import TimedRotatingFileHandler

os.makedirs("gen/logs/", exist_ok=True)
os.makedirs("gen/models/", exist_ok=True)

file_handler = TimedRotatingFileHandler("gen/logs/project-genesis.log", when='midnight', backupCount=7)
file_handler.namer = lambda name: name.replace(".log", "") + ".log"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
    handlers=[file_handler, console_handler]
)