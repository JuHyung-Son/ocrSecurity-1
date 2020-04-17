from urllib.request import urlopen, urlretrieve
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import os
import requests as req
import re
def ImageDownload(curdir, color_):
    #print ("sub directory : {}".format(curdir))
    # 디렉토리 이동
    f = re.compile('^https')
    os.chdir(curdir)
    print ("작업 위치 {}".format(os.getcwd()))
    f = re.compile('^https')
    data = {
        'q': color_,
        'source': 'lnms',
        'tbm': 'isch'
    }
    html = req.get(url="https://www.google.co.kr/search?", params=data)
    html = html.text
    bsObject = BeautifulSoup(html, "html.parser")
    a_Tag = bsObject.select("img")

    cnt = 1
    for indx, i in enumerate(a_Tag):
        if f.search(i.attrs['src']):
            print(indx + 1, i.attrs['src'])
            urlretrieve(i.attrs['src'], str(cnt) + "_" +".jpg")
            cnt += 1
    print ("============================================================")


