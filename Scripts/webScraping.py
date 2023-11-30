import requests
from bs4 import BeautifulSoup
import random
from urllib.parse import urljoin, urlsplit, SplitResult
import re

class WebScraper:
    def __init__(self, url, agents = None):
        self.url = url
        self.agents = agents 
        self.html_contet = self.fetch_html()
        self._soup = self._get_soup()

    def _get_soup(self):
        if self.html_contet is not None:
            return BeautifulSoup(self.html_contet, 'html.parser')
        else:
            ValueError("No HTML content to extract data from.")
            return None

    def fetch_html(self):
        try:
            user_agent = {'User-Agent': random.choice(self.agents)} if self.agents else {}
            return requests.get(self.url, headers=user_agent).text
        except requests.HTTPError as http_err:
            if http_err.response.status_code == 404:
                ValueError("404 Error: Page Not Found")
                return None
            else:
                ValueError(f"HTTP Error occurred: {http_err}")
                return None
        except Exception as err:
            ValueError(f'An error occurred: {err}')
            return None

    def find_all(self, get = None, tag=None, class_=None, **kwargs):
        if self._soup:
            if get == None:
                return [content for content in self._soup.find_all(tag, class_, **kwargs)]
            elif get == 'text':
                return [content.text for content in self._soup.find_all(tag, class_, **kwargs)]
            elif get == 'content':
                return [content.get('content') for content in self._soup.find_all(tag, class_, **kwargs)]
            else:
                raise ValueError("Argument get = {get} is not a valid argument")
        else:
            ValueError("No soup content")
            return None

    def select(self, selector, get = 'text'):
        if self._soup:
            if get == 'text':
                return [content.text for content in self._soup.select(selector)]
            elif get == 'content':
                return [content.get('content') for content in self._soup.select(selector)]
            else:
                raise ValueError("Argument get = {get} is not a valid argument")
        else:
            ValueError("No soup content")
            return None
        
    def find(self, get = 'text', tag=None, class_=None, **kwargs):
        if self._soup:
            if get == 'text':
                return [content.text for content in self._soup.find(tag, class_, **kwargs)]
            elif get == 'content':
                return [content.get('content') for content in self._soup.find(tag, class_, **kwargs)]
            else:
                raise ValueError("Argument get = {get} is not a valid argument")
        else:
            ValueError("No soup content")
            return None
        
    def scrape(self, method='find_all', get = 'text', *args, **kwargs):
        if method == 'find_all':
            return self.find_all(get = get, *args, **kwargs)
        elif method == 'select':
            return self.select(get = get, *args, **kwargs)
        elif method == 'find':
            return self.find(get = get, *args, **kwargs)
        else:
            raise ValueError("Invalid scraping method. Use 'find_all', 'select', or 'find'.")

class GetURLS(WebScraper):
    def __init__(self, url, agents = None):
        super().__init__(url, agents)
        self.domain = urlsplit(self.url).netloc
        self.urls = set()
    
    def preprocess_url(self, referrer, url):
        if not url:
            return None
        fields = urlsplit(urljoin(referrer, url))._asdict() 
        fields['path'] = re.sub(r'/$', '', fields['path']) 
        fields['fragment'] = ''
        fields = SplitResult(**fields)
        if fields.netloc == self.domain:
            if fields.scheme == 'http':
                httpurl = cleanurl = fields.geturl()
                httpsurl = httpurl.replace('http:', 'https:', 1)
            else:
                httpsurl = cleanurl = fields.geturl()
                httpurl = httpsurl.replace('https:', 'http:', 1)
            if httpurl not in self.urls and httpsurl not in self.urls:
                return cleanurl
        return None
    
    def scrape_recursive(self, url=None):
        ''' Scrape the URL and its outward links in a depth-first order.
            If URL argument is None, start from main page.
        '''
        if url is None:
            url = self.url

        print("Scraping {:s} ...".format(url))
        self.urls.add(url)
        for link in self._soup.findAll("a"):
            childurl = self.preprocess_url(url, link.get("href"))
            if childurl:
                self.scrape_recursive(childurl)
