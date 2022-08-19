import logging
import datetime
# 设置logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',level=logging.DEBUG)
logger = logging.getLogger(__name__)

cur_time = datetime.datetime.now()
filehandler = logging.FileHandler("log/"+cur_time.strftime('%y%m%d%H%M')+"log.txt")
filehandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
filehandler.setLevel(logging.INFO) 
logger.addHandler(filehandler)