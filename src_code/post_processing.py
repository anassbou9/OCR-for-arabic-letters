import os
import urllib3
import io
import gzip
import sys
import re
import urllib.request

from urllib.parse import quote
from bs4 import BeautifulSoup
from io import BytesIO
from pyarabic.araby import tokenize


def getPage(url):
    request = urllib.request.Request(url)
    request.add_header("Accept-encoding", "gzip")
    request.add_header(
        "User-Agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    )
    response = urllib.request.urlopen(request)
    if response.info().get("Content-Encoding") == "gzip":
        buf = BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = f.read()
    else:
        data = response.read()
    return data


def didYouMean(q):
    q = str(str.lower(q)).strip()
    url = "http://www.google.com/search?q=" + quote(q)
    html = getPage(url)
    soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")
    try:
        ans = soup.find("b").get_text(strip=True)
    except:
        ans = 1
    return ans


def post_process(text):
    tokens = tokenize(text)
    sentences = [tokens[x : x + 5] for x in range(0, len(tokens), 5)]

    n = len(sentences)
    output_text = []
    for i in range(n):
        sentence = " ".join(sentences[i])
        result = didYouMean(sentence)
        if result == 1:
            output_text.append(sentence)
        else:
            output_text.append(result)

    return " ".join(output_text)


if __name__ == "__main__":
    print(
        post_process(
            "اللُّغَةُ العَرَبِيَّة هي أكثر اللغات السامية تحدسا، وإحدى أكثر اللغات انتشاراً في العالم، يتحدثها أكثر من 467 مليون نسمة"
        )
    )
