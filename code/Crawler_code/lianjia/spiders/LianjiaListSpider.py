import json
import sqlite3

import scrapy
import unicodecsv as csv


class LianjiaListSpider(scrapy.Spider):
    name = "LianjiaList"

    def start_requests(self):
        with open("location.csv", mode='rb') as f:
            csv_file = csv.reader(f, encoding='utf-8')
            for row in csv_file:
                yield scrapy.Request(url=row[2], callback=self.parse)

    def parse(self, response):
        district = response.xpath("/html/body/div[3]/div/div[1]/dl[2]/dd/div[1]/div[1]").css("a.selected::text").get()
        location = response.xpath("/html/body/div[3]/div/div[1]/dl[2]/dd/div[1]/div[2]").css("a.selected::text").get()
        loc_url = response.xpath("/html/body/div[3]/div/div[1]/dl[2]/dd/div[1]/div[2]").css(
            "a.selected::attr(href)").get()

        print(location, response.url)

        conn = sqlite3.connect("lianjia.db")

        conn.execute("CREATE TABLE IF NOT EXISTS house("
                     "url text UNIQUE,"
                     "community text,"
                     "district text,"
                     "location text,"
                     "house_info text,"
                     "position_info text,"
                     "unit_price text,"
                     "total_price text,"
                     "title text);")

        items = response.css(".sellListContent li .info")
        for item in items:
            title = item.css(" .title a::text").get()
            url = item.css(" .title a::attr(href)").get()
            community = item.css(" .address .houseInfo a::text").get()
            house_info = item.css(" .address .houseInfo::text").get()
            position_info = item.css(".flood .positionInfo::text").get()
            total_price = item.css(" .priceInfo .totalPrice span::text").get()
            unit_price = item.css(" .priceInfo .unitPrice span::text").get()
            try:
                conn.execute("INSERT INTO house VALUES(?,?,?,?,?,?,?,?,?)",
                             (url, community, district, location, house_info, position_info, unit_price, total_price,
                              title))
                conn.commit()
            except sqlite3.IntegrityError:
                return
        try:
            page_data_text = response.css(
                ".content .leftContent .contentBottom .page-box::attr(page-data)"
            ).get()
            if page_data_text is not None:

                page_data = json.loads(page_data_text)
                cur_page = page_data['curPage']
                if cur_page < page_data['totalPage']:
                    new_url = "https://sh.lianjia.com" + loc_url + "pg" + str(cur_page + 1) + "/"
                    yield scrapy.Request(url=new_url, callback=self.parse)
        except BaseException as e:
            print(e)
