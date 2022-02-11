import datetime
import random
from xmlrpc.client import DateTime

import pymysql
import requests
from bs4 import BeautifulSoup
import app.models as models
from app import db
import time
import _thread
import re
import csv
from selenium.webdriver import Chrome
from selenium.webdriver import ChromeOptions



'''
i=1
prefix='http://example.python-scraping.com'

url = 
r = requests.get(url)
print(r.status_code)
#乱码转换成二进制码再解码成字符串形式
html = r.text.encode(r.encoding).decode()
soup = BeautifulSoup(html,'lxml')
print(soup)
'''
from selenium import webdriver
# 在单词表里用index记录文档，提取文档时用url_dic[urls[index]][0]

import os

def request_page(url):
    headers = {'accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
               'accept-encoding': 'gzip, deflate, br',
               'referer':'https://rateyourmusic.com/',
               'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'}

    r = requests.get(url,headers=headers)
    print(r.status_code)
    # 乱码转换成二进制码再解码成字符串形式
    html = r.text.encode(r.encoding).decode()
    soup = BeautifulSoup(html, 'lxml')
    return soup

def request_page_with_webdriver(url, website):
    chrome_driver = 'C:/Program Files/Google/Chrome/Application/chromedriver.exe'  # chromedriver的文件位置
    option = ChromeOptions()
    option.add_argument("--disable-blink-features=AutomationControlled")
    browser = webdriver.Chrome(executable_path=chrome_driver,options=option)
    browser.implicitly_wait(5)
    browser.get(url)
    if website=="rateyourmusic":
        browser.find_element_by_class_name("fc-button.fc-cta-consent.fc-primary-button").click()
        js = "var q=document.documentElement.scrollTop=700"
        browser.execute_script(js)
    time.sleep(5)
    html = browser.execute_script("return document.documentElement.outerHTML")
    soup = BeautifulSoup(html, 'lxml')
    browser.quit()
    return soup



def search_url_parse(keywords, strcode, website):
    url=""
    if website == "musicbrainz":
        if strcode == "010":
             url="https://musicbrainz.org/search?query=" + \
             keywords["artist"].replace(" ","+")+"&type=artist&method=indexed"
    if website == "rateyourmusic":
        if strcode == "110":
            album=keywords["artist"]
            album_name=album.split(' ')
            album_term='%20'.join(album_name)
            url="https://rateyourmusic.com/search?searchterm=" + \
                album_term+"&searchtype="
    return url


def soup_search_url_extract(strcode, website ,url):
    request_page_with_webdriver(url,"musicbrainz")
    soup = request_page(url)
    url = ""
    if website == "musicbrainz":
        if strcode == "010":
            soup=soup.find('table', class_='tbl')
            soup=soup.find('tr', class_="odd")
            url="https://musicbrainz.org/"+soup.find_all('a', href=True)[0]['href']
    return url

def musicbrainz_artist_relationship_extract(relation_url):
    relationSoup = request_page(relation_url)
    key_info = {"members:": 0, "member of:": 0, "tribute artists:": 0, "tribute to": 0, "teachers:": 0, "students:": 0,
                "tours:": 0, "part of:": 0, }
    tables = relationSoup.find_all('table', class_="details")
    for table in tables:
        trs = table.find_all("tr")
        for tr in trs:
            if (tr.find("th").get_text()) in key_info.keys():
                info_list = []
                for a in tr.find("td").find_all("a"):
                    info_list.append(a.find("bdi").get_text())
                key_info[tr.find("th").get_text()] = info_list
    return key_info
def musicbrainz_album_relationship_extract(relation_url):
    relationSoup = request_page(relation_url)
    key_info = {"part of:": 0, "Allmusic:": 0 }
    tables = relationSoup.find_all('table', class_="details")
    for table in tables:
        trs = table.find_all("tr")
        for tr in trs:
            if (tr.find("th").get_text()) in key_info.keys():
                info_list = []
                if tr.find("th").get_text()=="Allmusic:":
                    info_list.append(tr.find("td").find("a").find("bdi").get_text())
                else:
                    for a in tr.find("td").find_all("a"):
                        info_list.append(a.find("bdi").get_text())
                key_info[tr.find("th").get_text()] = info_list
    return key_info
def musicbrainz_artist_tag_extract(tags_url):
    tagSoup = request_page(tags_url)

    ul = tagSoup.find("ul", class_="genre-list")
    tagDic = {}
    lis = ul.find_all("li")
    for li in lis:
        tagDic[li.find("a").get_text()] = li.find("span", class_="tag-count").get_text()
    return tagDic

def musicbrainz_wiki_content_extract(soup_page):
    wiki = soup_page.find('div', class_=["wikipedia-extract-body", "wikipedia-extract-collapsed"])
    wiki_list = []
    if wiki is not None:
        paragraphs = wiki.find_all('p')
        for i in paragraphs:
            wiki_list.append(i.get_text())
    wiki_content = ' '.join(wiki_list)
    return wiki_content

def musicbrainz_album_url_extract(soup_page):
    table=soup_page.find("table", class_=["tbl" , "release-group-list"])
    trs=table.find_all("tr")
    print(trs)
    url_dic={}
    for tr in trs[1:]:
        year=tr.find("td", class_="c").get_text()
        url=tr.find("a")["href"]
        if year!="—":
            url_dic[url]=year

    new_url_dic = {}
    if len(url_dic)>6:

        random_albums = random.sample(list(url_dic), 6)
        for i in random_albums:
            new_url_dic[i]=url_dic[i]
        url_dic=new_url_dic
    return url_dic

def musicbrainz_album_songs_info_extract(url):
    soup = request_page(url)

def musicbrainz_album_info_extract(url_dic):
    for i in url_dic.keys():

        url="https://musicbrainz.org/"+i
        request_page_with_webdriver(url, "musicbrainz")
        soup=request_page(url)
        wiki_content=musicbrainz_wiki_content_extract(soup)
        tabs = soup.find('ul', class_='tabs')
        tabs = tabs.find_all('a')
        tags_url = "https://musicbrainz.org/" + tabs[2]['href']
        tagDic = musicbrainz_artist_tag_extract(tags_url)
        table = soup.find("table", class_="tbl")
        trs = table.find_all("tr")

        links=trs[2].find("td").find_all("a")

        keywords = {"album": 0, "artist": 0, "single": 0}
        for i in links:
            if i.find("bdi")!=None:
                keywords["album"]=i.find("bdi").get_text()
                musicbrainz_album_songs_info_extract("https://musicbrainz.org/"+i["href"])


        keywords["artist"]=trs[2].find_all("td")[1].find("bdi").get_text()

        website = ["rateyourmusic"]
        strcode = ""
        for p in keywords.keys():
            if keywords[p] == 0:
                strcode += "0"
            else:
                strcode += "1"
        url=search_url_parse(keywords,strcode, website[0])
        score, rater_number, rating, review = rateyourmusic_soup_page_info_extract(strcode,keywords,url)

def rateyourmusic_reviews_extract(url):
    soup =request_page_with_webdriver(url, "rateyourmusic")
    time.sleep(2)
     #request_page(url)

    ratings = soup.find_all("span", attrs={"itemprop":"ratingValue"})
    reviews= soup.find_all("div", class_="review_body")
    rating=[]
    review=[]
    for i in range(len(ratings)):
        rating.append(ratings[i].attrs["content"])
        review.append(reviews[i].find("span", class_="rendered_text").get_text())
    return rating, review


def rateyourmusic_soup_page_info_extract(strcode,keywords,url):
    if strcode == "110":
        info_dic={"score":0, "rate_number":0, "reviews":0}
        request_page_with_webdriver(url, "rateyourmusic")
        time.sleep(2)
        soup = request_page(url)
        time.sleep(5)
        infobox=soup.find("tr", class_="infobox")
        artist_url="https://rateyourmusic.com/"+infobox.find("a")["href"]
        soup = request_page(artist_url)
        discos=soup.find("div", id="disco_type_s").find_all("div", class_="disco_release")
        for disco in discos:
            link=disco.find("div", class_="disco_info").find("div", class_="disco_mainline").find("a")
            if link.get_text().lower()==keywords["album"].lower():
                score=disco.find("div", class_=["disco_avg_rating","enough_data"]).get_text()
                rater_number=disco.find("div", class_=["disco_ratings"]).get_text()
                info_dic["score"]=score
                info_dic["rate_number"]=rater_number
                url="https://rateyourmusic.com"+link["href"]
                time.sleep(2)
                rating, review=rateyourmusic_reviews_extract(url)
        return score, rater_number, rating, review




def musicbrainz_soup_page_info_extract(strcode , url):
    if strcode == "010":
        request_page_with_webdriver(url, "musicbrainz")
        soup = request_page(url)
        url_dic = musicbrainz_album_url_extract(soup)
        musicbrainz_album_info_extract(url_dic)
        property = soup.find('dl', class_='properties')
        name = property.find('dd', class_='sort-name').get_text()
        gender = property.find('dd', class_='gender').get_text()
        born = property.find('dd', class_='begin-date').get_text()
        born = born.split(' ')[0]
        born = datetime.date(*map(int, born.split('-')))
        print(name, gender, born)
        artist_query=models.Artist.query.filter_by(artist_name=name, gender=gender, Born=born).all()
        artist=[]
        if len(artist_query) != 0:
            return

        wiki_content = musicbrainz_wiki_content_extract(soup)
        tabs = soup.find('ul', class_='tabs')
        tabs = tabs.find_all('a')
        relation_url = "https://musicbrainz.org/" + tabs[5]['href']
        key_info = musicbrainz_artist_relationship_extract(relation_url)
        tags_url = "https://musicbrainz.org/" + tabs[7]['href']
        tagDic = musicbrainz_artist_tag_extract(tags_url)

        artist = models.Artist(artist_name=name, gender=gender, Born=born, wikipedia=wiki_content, tags=str(tagDic))
        db.session.add(artist)
        db.session.commit()
        if key_info["member of:"] != 0:
            for i in key_info["member of:"]:
                band = None
                if len(models.Bands.query.filter_by(band_name=i).all()) == 0:
                    band = models.Bands(band_name=i)
                    db.session.add(band)
                else:
                    band = models.Bands.query.filter_by(band_name=i).first()
                artist.band.append(band)
                db.session.commit()

        if key_info["tours:"] != 0:
            for i in key_info["tours:"]:
                tour = None
                if len(models.Tours.query.filter_by(tour_name=i).all()) == 0:
                    tour = models.Tours(tour_name=i)
                    db.session.add(tour)
                else:
                    tour = models.Tours.query.filter_by(tour_name=i).first()
                artist.tour.append(tour)
                db.session.commit()

        if key_info["part of:"] != 0:
            for i in key_info["part of:"]:
                nominate = None
                if len(models.Nominates.query.filter_by(nominate_name=i).all()) == 0:
                    nominate = models.Nominates(nominate_name=i)
                    db.session.add(nominate)
                else:
                    nominate = models.Nominates.query.filter_by(nominate_name=i).first()
                artist.nominate.append(nominate)
                db.session.commit()

if __name__ == "__main__":
    keywords={"album":0, "artist":"taylor swift", "single":0}
    website=["musicbrainz"]
    strcode=""
    for p in keywords.keys():
        if keywords[p]==0:
            strcode+="0"
        else:
            strcode+="1"

    url=search_url_parse(keywords, strcode, website[0])
    # 打开数据库连接
    '''
    db = pymysql.connect(host="localhost",user ="root", password ="Q13579qscesz", database ="musicdb", charset ="utf8")

    cursor = db.cursor()
    
    insert_sql =
    # 执行sql语句
    cursor.execute("insert into studys(id, name, age) values(3, '骑着乌龟赶猪', 35)")
    # 提交到数据库执行
    db.commit()
    cursor.execute("select * from studys")
    # 查看表里所有数据
    data = cursor.fetchall()
    print(data)  # 关闭数据库连接 db.close()
    '''
    url=soup_search_url_extract(strcode, website[0], url)
    musicbrainz_soup_page_info_extract(strcode, url)
    print(str([1,2,3,4]))
'''
def process():
    test = 334
    bb = os.getcwd()
    chrome_driver = 'C:/Program Files/Google/Chrome/Application/chromedriver.exe'  # chromedriver的文件位置
    while test >= 0:
        test += 1
        # url='https://asiantolick.com/post-'+str(test)
        url = 'https://hentai.ex-panda.workers.dev/0:/PANDA/'
        b = webdriver.Chrome(executable_path=chrome_driver)

        b.get(url)

        time.sl eep(20)
        b.get(url)
        time.sleep(5)
        aa = b.find_elements_by_class_name("fa.fa-download.faa-shake.animated-hover")
        x = 48
        for i in aa:
            i.click()
            js = "var q=document.documentElement.scrollTop=" + str(x)
            x += 48
            b.execute_script(js)
            time.sleep(23)
        
        aa=b.find_element_by_class_name("download_post")
        b.get(aa.get_attribute('href'))
        js = "var q=document.documentElement.scrollTop=400"
        b.execute_script(js)

        bb=b.find_element_by_class_name("download_post")
        bb.click()
        time.sleep(23)
        html_text = b.page_source
        

        b.quit()



        os.mkdir(bb + '\\' + str(test))
        disk = bb + '\\' + str(test)



        soup = BeautifulSoup(html_text,'lxml')

        soup=soup.find('div', class_='spotlight-group')

        ind = 0
        j=soup.find_all('img')
        for i in j:
            ind+=1
            try:
                r = requests.get(i['src'].replace('/thumb',''), stream=True, timeout=5)
                print(i['src'])
                print(r.status_code)  # 返回状态码
                if r.status_code == 200:
                    open(disk+'\\img'+str(ind)+'.jpg', 'wb').write(r.content)  # 将内容写入图片
                    print("done")
                r.close()
            except:
                pass
            time.sleep(3)
'''



'''
#print(soup.find_all(href=re.compile("/places/default/")))
urls=['http://example.python-scraping.com/places/default/index/1']
htmls={}
for i in urls:
    crawl=0
    r = requests.get(i)
    html = r.text.encode(r.encoding).decode()
    soup = BeautifulSoup(html,'lxml')
    time.sleep(5)
    for i in soup.find_all('a'):
       url='http://example.python-scraping.com'+i['href']
       if url not in urls:
           urls.append(url)


# 为线程定义一个函数

def request( threadName, delay):


      time.sleep(delay)
      count += 1
      print ("%s: %s" % ( threadName, time.ctime(time.time()) ))

# 创建两个线程
try:
   _thread.start_new_thread( request, ("Thread-1", 5, ) )
   _thread.start_new_thread( print_time, ("Thread-2", 4, ) )
except:
   print ("Error: 无法启动线程")

while 1:
   pass
'''
