import json,re
from requests_html import HTMLSession
import pandas as pd
session = HTMLSession()

r = session.get('https://www.tripadvisor.com.vn/Hotels-g293921-Vietnam-Hotels.html')
