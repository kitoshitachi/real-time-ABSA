from requests_html import HTMLSession

session = HTMLSession()

headers = {
    "accept": "text/html, */*",
    "accept-language": "vi,en-US;q=0.9,en;q=0.8",
    "cache-control": "no-cache",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "pragma": "no-cache",
    "sec-ch-ua": "\"Google Chrome\";v=\"107\", \"Chromium\";v=\"107\", \"Not=A?Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-puid": "35fcf35b-3350-4be8-ab2a-9d999f2da60a",
    "x-requested-with": "XMLHttpRequest"
}

r = session.post("https://www.tripadvisor.com.vn/Hotels-g293921-Vietnam-Hotels.html", headers=headers,
                 json="plSeed=94665279&offset=0&limit=100&reqNum=2&isLastPoll=false&paramSeqId=7&waitTime=4261&changeSet=&puid=35fcf35b-3350-4be8-ab2a-9d999f2da60a")

print(r.text)
