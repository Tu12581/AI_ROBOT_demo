import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import random
import re
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm
import logging as lg

"""网页爬取"""

def Creeper(key_word, max_pages=2):
    """
    爬取百度keyword关键词搜索结果
    maxpage: 表示爬取搜索结果页数
    返回: dataframe
    """

    # 分别代表dataframe的列
    page_list = []
    kw_list = []
    title_list = []
    href_list = []
    real_url_list = []
    desc_list = []
    site_list = []
    # 开始爬取
    for page in range(max_pages):
        lg.info("百度搜索：开始爬取第{}页".format(page+1))
        wait_seconds = random.uniform(1, 2)
        lg.info("百度搜索：等待{}秒".format(wait_seconds))
        sleep(wait_seconds)

        url = "http://www.baidu.com/s?wd=" + key_word + "&pn=" + str(page*10)
        r = requests.get(url, headers=header)
        html = r.text
        lg.info("百度搜索：响应码为{}".format(r.status_code))

        # 根据返回的html文本进行解析，过滤出搜索结果
        soup = BeautifulSoup(html, "html.parser")
        result_list_0 = soup.find_all(class_='result c-container new-pmd')
        result_list_1 = soup.find_all(class_='result c-container xpath-log new-pmd')
        result_list = result_list_0 + result_list_1
        lg.info("百度搜索：正在爬取{}， 共查询到{}个结果".format(url, len(result_list)))

        # 对于所有的搜索结果分别获取其标题及简介
        for result in result_list:
            title = result.find('a').text
            # print("title is:", title)
            href = result.find('a')["href"]
            real_url = get_real_url(v_url=href)

            try:
                desc = result.find(class_="content-right_8Zs40").text
            except:
                desc = ""
            try:
                site = result.find(class_="c-color-gray").text
            except:
                site = ""
            # 记录每条结果
            kw_list.append(key_word)
            page_list.append(page + 1)
            title_list.append(title)
            href_list.append(href)
            real_url_list.append(real_url)
            desc_list.append(desc)
            site_list.append(site)
    df = pd.DataFrame(
        {
            "关键词": kw_list,
            "页码": page_list,
            "标题": title_list,
            "简介": desc_list,
            "百度链接": href_list,
            "真实链接": real_url_list,
            "网站名称": site_list
        }
    )
    return df

def get_real_url(v_url):
    """
    根据跳转链接获取真实连接
    v_url: 跳转链接
    real_url: 真实链接
    """
    r = requests.get(v_url, headers=header, allow_redirects=False)
    if r.status_code == 302:
        real_url = r.headers.get("Location")
    else:
        real_url = re.findall("URL='(.*?)'", r.text)[0]
    return real_url

# 取得网页html
def get_html(site):
    """
    调用selenium获取site的html源码
    """
    s = Service('msedgedriver.exe')
    # 我们并不需要浏览器弹出
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    # 启动浏览器的无头模式，访问
    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(10)
    try:
        driver.get(site)
    except TimeoutException as e:
        lg.error("time out:", e)

    # 获取页面的源代码
    page_source = driver.page_source
    driver.quit()
    return page_source


def parser(soup):
    """
    给定html文本对象soup
    返回其正文部分
    """
    # 用于保存html对象
    result_list = []
    # 百家号、搜狐专属--------------------------------------------------------------------------------------------------------
    try:
        bjh = soup.find_all(name='div', attrs={'data-testid': 'article'})[0]  # 使用能够覆盖正文的tag，只会找到一个，所以直接索引0。
        result_list += bjh.find_all("p")
        result_list += bjh.find_all("span")
    except:
        pass
    try:
        bjh = soup.find_all("article")[0]
        result_list += bjh.find_all("p")
        result_list += bjh.find_all("span")
    except:
        pass
    # 澎湃专属--------------------------------------------------------------------------------------------------------
    try:
        pp = soup.find_all(name="div", attrs={"class": "index_wrapper__L_zqV"})[0]
        result_list += pp.find_all("p")
        result_list += pp.find_all("span")
    except:
        pass
    # 163专属--------------------------------------------------------------------------------------------------------
    try:
        _163 = soup.find_all(name="div", attrs={"class": "post_body"})[0]
        result_list += _163.find_all("p")
    except:
        pass
    # 站长专属--------------------------------------------------------------------------------------------------------
    try:
        zz = soup.find_all(name="div", attrs={"class": "pcontent", "id": "article-content"})[0]
        result_list += zz.find_all("p")
    except:
        pass
    # 腾讯新闻专属--------------------------------------------------------------------------------------------------------
    try:
        tx = soup.find_all(name="div", attrs={"class": "content-article"})[0]
        # pattern = re.compile("^qnt-")
        result_list += tx.find_all(class_="qnt-p")
    except:
        pass
        # 新浪专属--------------------------------------------------------------------------------------------------------
    try:
        xl = soup.find_all(name="div", attrs={"class": "article", "id": "artibody"})[0]
        result_list += xl.find_all("p")
    except:
        pass
        # b站专属--------------------------------------------------------------------------------------------------------
    try:
        bz = soup.find_all(name="div", attrs={"class": "article-container"})[0]
        result_list += bz.find_all("p")
    except:
        pass
        # 凤凰新闻专属--------------------------------------------------------------------------------------------------------
    try:
        fh = soup.find_all(name="div", attrs={"class": "index_main_content_j-HoG"})[0]
        result_list += fh.find_all("p")
    except:
        pass
        # 若上述皆未找到--------------------------------------------------------------------------------------------------------
    if len(result_list) == 0:
        pattern = re.compile(".*content.*")
        result_list += soup.find_all(pattern)
        pattern = re.compile(".*article.*")
        result_list += soup.find_all(pattern)
    # --------------------------------------------------------------------------------------------------------

    if len(result_list) == 0:
        lg.error("解析器:未发现正文部分")
    else:
        lg.info("解析器：共解析到{}个结果".format(len(result_list)))
    # 对于每个html对象，获取其文本并拼接
    res_body = ""
    for res in result_list:
        res_body += res.text
    return res_body


# 使用header作为头，用于爬虫访问网站
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.46",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Connection": "keep-alive",
    "Host": "www.baidu.com",
    # 一般cookie要常换
    "Cookie": "BIDUPSID=B74BEE28F591253CB775FB0142D52FC2; PSTM=1689585249; BAIDUID=B74BEE28F591253CF56870F22EFCC449:FG=1; BAIDUID_BFESS=B74BEE28F591253CF56870F22EFCC449:FG=1; COOKIE_SESSION=34928_1_6_7_7_3_1_0_6_2_0_0_34832_0_4_0_1694389091_1690388333_1694389087%7C9%23735794_9_1690388331%7C6; BAIDU_WISE_UID=wapp_1694430439254_663; ZFY=Bk0QHaMv6sWqRVWER3GYjzk0SUaJaWd4P20GIQMqBrw:C; BD_UPN=12314753; BA_HECTOR=008l210l04al0la180ag0k041ij5a1k1o; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BD_CK_SAM=1; PSINO=2; H_PS_PSSID=39318_38831_39399_39396_39419_39535_39438_39540_39497_39234_39466_26350_39422; delPer=0; baikeVisitId=cdc30dd1-51a7-4620-beec-d2f7c9f0a1f5; H_PS_645EC=fd50i1NCbX7z4hb86lJCLoimNedSK%2FqhjkCuMLyGssdVMq7QnjmYEdTuhJU; BDSVRTM=171"
}

def get_llm_url(key_word):  # 仅用于获取相关的url
    df = Creeper(key_word)
    site_list = df["真实链接"].tolist()
    return site_list[:10]  #获取前十个真实链接用于后续爬取


def get_text(key_word):
    log = lg.basicConfig(level=lg.INFO,
                         format="%(asctime)s - %(levelname)s - %(message)s",
                         filename="log.log",
                         filemode="w")
    df = Creeper(key_word)
    site_list = df["真实链接"].tolist()
    # 解析并爬取真实链接
    body_list = []
    texts = {}  # 存储爬下来的内容
    for url in tqdm(site_list):
        lg.info("正在爬取{}， ".format(url))
        html = get_html(url)
        soup = BeautifulSoup(html, "html.parser")

        res_body = parser(soup)
        if res_body == '':
            res_body = '没有得到结果'  # 未得到body则不存储
        else:
            texts[f"{url}"] = res_body
        body_list.append(res_body)
    return texts

if __name__=='__main__':
    # 注册表进行监控
    log = lg.basicConfig(level=lg.INFO,
                         format="%(asctime)s - %(levelname)s - %(message)s",
                         filename="log.log",
                         filemode="w")
    key_word = "陈科海"
    df = Creeper(key_word)
    site_list = df["真实链接"].tolist()

    # 解析并爬取真实链接
    body_list = []
    texts = {}  # 存储爬下来的内容
    for url in tqdm(site_list):
        lg.info("正在爬取{}， ".format(url))
        html = get_html(url)
        soup = BeautifulSoup(html, "html.parser")

        res_body = parser(soup)
        if res_body == '':
            res_body = '没有得到结果'  # 未得到body则不存储
        else:
            texts[f"{url}"] = res_body
        body_list.append(res_body)

    df["body"] = body_list
    df.to_csv(key_word + ".csv", encoding="utf_8_sig", index=False)
