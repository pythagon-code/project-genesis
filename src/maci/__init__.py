import os
import logging
from logging.handlers import TimedRotatingFileHandler

if os.path.commonpath([os.getcwd(), __file__]) == os.path.dirname(__file__):
    raise Exception("Don't run the project inside project directory.")

os.makedirs("logs/", exist_ok=True)
os.makedirs("models/", exist_ok=True)

file_handler = TimedRotatingFileHandler("logs/maci.log", when='midnight', backupCount=7)
file_handler.namer = lambda name: name.replace(".log", "") + ".log"

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
    handlers=[file_handler, console_handler]
)