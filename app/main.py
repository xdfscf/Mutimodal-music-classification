import requests
from bs4 import BeautifulSoup
import time
import _thread
import re
import csv
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
#在单词表里用index记录文档，提取文档时用url_dic[urls[index]][0]

import os
def process():
    test = 334
    bb = os.getcwd()
    chrome_driver = 'C:/Program Files/Google/Chrome/Application/chromedriver.exe'  # chromedriver的文件位置
    while test>=0:
        test+=1
        #url='https://asiantolick.com/post-'+str(test)
        url='https://hentai.ex-panda.workers.dev/0:/PANDA/'
        b = webdriver.Chrome(executable_path=chrome_driver)

        b.get(url)

        time.sleep(20)
        b.get(url)
        time.sleep(5)
        aa=b.find_elements_by_class_name("fa.fa-download.faa-shake.animated-hover")
        x=48+194*48
        for i in aa[194:]:

            i.click()
            js = "var q=document.documentElement.scrollTop="+str(x)
            x+=48
            b.execute_script(js)
            time.sleep(23)
        '''
        aa=b.find_element_by_class_name("download_post")
        b.get(aa.get_attribute('href'))
        js = "var q=document.documentElement.scrollTop=400"
        b.execute_script(js)

        bb=b.find_element_by_class_name("download_post")
        bb.click()
        time.sleep(23)
        html_text = b.page_source
        '''

        b.quit()
'''
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

process()

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
