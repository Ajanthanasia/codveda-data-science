import requests
from bs4 import BeautifulSoup
import pandas

def webScrapingDataFunc(url):
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, "html.parser")

    quotes_data = []
    
    quotes = soup.find_all("div", class_="quote")
    
    for q in quotes:
        text = q.find("span", class_="text").text
        author = q.find("small", class_="author").text

        quotes_data.append({
            "id": len(quotes_data) + 1,
            "quote": text,
            "author": author,
            "tags": [tag.text for tag in q.find_all("a", class_="tag")]
        })
        
        details = pandas.DataFrame(quotes_data)
        
    # print(details)
    return details

#init website link
url = "https://quotes.toscrape.com"

detailsData = webScrapingDataFunc(url)

detailsData.to_csv("quotes.csv", index=False)