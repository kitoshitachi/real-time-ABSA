import time




import os
cmd = 'start cmd /k python'
os.system(f"{cmd} topic.py")
time.sleep(5)
os.system(f"{cmd} scraping.py 293925")
time.sleep(5)
os.system(f"{cmd} worker.py")



