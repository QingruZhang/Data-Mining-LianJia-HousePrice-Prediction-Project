import json
import re
import sqlite3

import scrapy
import unicodecsv as csv

from model import *


class LianjiaListSpider(scrapy.Spider):
    name = "Lianjia"

    def start_requests(self):
        urls = []
        with open("shanghai.csv", mode='rb') as f:
            csv_file = csv.reader(f, encoding='utf-8')
            for row in csv_file:
                urls.append(row[0])

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        url = response.url
        title = response.css(".sellDetailHeader .title-wrapper .content .title .main::text").get()
        subtitle = response.css(".sellDetailHeader .title-wrapper .content .title .sub::text").get()
        total_price = response.css(".overview .content .price .total::text").get()
        unit_price = response.css(".overview .content .price .text .unitPrice .unitPriceValue::text").get()
        community = response.css(".overview .content .aroundInfo .communityName .info::text").get()
        community_url = "https://sh.lianjia.com" + response.css(
            ".overview .content .aroundInfo .communityName .info::attr(href)").get()
        district = response.xpath("/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[1]/text()").get()
        location = response.xpath("/html/body/div[5]/div[2]/div[4]/div[2]/span[2]/a[2]/text()").get()

        huxing = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[1]/text()").get()
        floor = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[2]/text()").get()
        building_area = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[3]/text()").get()
        huxing_type = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[4]/text()").get()
        available_area = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[5]/text()").get()
        building_structure = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[6]/text()").get()
        face_to = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[7]/text()").get()
        building_type = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[8]/text()").get()
        decoration = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[9]/text()").get()
        elevator_rate = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[10]/text()").get()
        elevator = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[11]/text()").get()
        property_year = response.xpath("//*[@id=\"introduction\"]/div/div/div[1]/div[2]/ul/li[12]/text()").get()

        trade_type = response.xpath("//*[@id=\"introduction\"]/div/div/div[2]/div[2]/ul/li[2]/span[2]/text()").get()
        house_usage = response.xpath("//*[@id=\"introduction\"]/div/div/div[2]/div[2]/ul/li[4]/span[2]/text()").get()
        limit_year = response.xpath("//*[@id=\"introduction\"]/div/div/div[2]/div[2]/ul/li[5]/span[2]/text()").get()
        property_belong = response.xpath(
            "//*[@id=\"introduction\"]/div/div/div[2]/div[2]/ul/li[6]/span[2]/text()").get()
        mortgage = response.xpath(
            "//*[@id=\"introduction\"]/div/div/div[2]/div[2]/ul/li[7]/span[2]/text()").get().strip()

        tags = response.css(".baseinform .introContent .tags .tag::text").getall()
        list_tags = []
        for tag in tags:
            t = tag.strip()
            if t != '':
                list_tags.append(t)

        features = response.css(" .baseinform .introContent .baseattribute")

        dic_features = []
        for feature in features:
            dic_features.append({feature.css(".name::text").get(): feature.css(".content::text").get().strip()})

        try:
            house = HouseDetail(url=url, unit_price=unit_price, total_price=total_price, building_area=building_area,
                                title=title, subtitle=subtitle, community=community, community_url=community_url,
                                district=district, location=location, huxing=huxing, huxing_type=huxing_type,
                                floor=floor, available_area=available_area, building_structure=building_structure,
                                face_to=face_to, building_type=building_type, decoration=decoration, elevator=elevator,
                                elevator_rate=elevator_rate, property_belong=property_belong,
                                property_year=property_year, trade_type=trade_type, house_usage=house_usage,
                                limit_year=limit_year, mortgage=mortgage,
                                tags=json.dumps(list_tags, ensure_ascii=False),
                                feature=json.dumps(dic_features, ensure_ascii=False))
            house.save()
        except sqlite3.IntegrityError:
            return
        # average_price = response.xpath(
        #    "//*[@id=\"resblockCardContainer\"]/div/div/div[2]/div/div[1]/span/text()").get().strip()
        # building_type = response.xpath("//*[@id=\"resblockCardContainer\"]/div/div/div[2]/div/div[3]/span/text()").get()
        # building_year = response.xpath("//*[@id=\"resblockCardContainer\"]/div/div/div[2]/div/div[2]/span/text()").get()
        # building_count = response.xpath(
        #     "//*[@id=\"resblockCardContainer\"]/div/div/div[2]/div/div[4]/span/text()").get()
        # huxing_count = response.xpath(
        #     "//*[@id=\"resblockCardContainer\"]/div/div/div[2]/div/div[5]/span/text()").get().strip()
        gps = re.findall(r"resblockPosition.*?',", response.text)
        lat = 0
        log = 0
        if len(gps) != 0:
            lat, log = gps[0][17:-1].replace("'", "").split(',')
        # comm = Community(community=community, url=community_url, average_price=average_price,
        #                  building_type=building_type, building_year=building_year, building_count=building_count,
        #                 huxing_count=huxing_count, latitude=lat, longitude=log)
        try:
            comm = Community(district=district, location=location, community=community, url=community_url, latitude=lat,
                             longitude=log)
            comm.save()
        except sqlite3.IntegrityError:
            return
