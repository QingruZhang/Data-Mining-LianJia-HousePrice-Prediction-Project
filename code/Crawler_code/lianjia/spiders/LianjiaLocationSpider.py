import scrapy
import unicodecsv as csv


class LianjiaLocationSpider(scrapy.Spider):
    name = "LianjiaLocation"

    def start_requests(self):
        url = "https://sh.lianjia.com/ershoufang/"
        districts = ['pudong', 'minhang', 'baoshan', 'xuhui', 'songjiang', 'jiading', 'jingan', 'putuo',
                     'yangpu', 'hongkou', 'changning', 'huangpu', 'qingpu', 'fengxian', 'jinshan', 'chongming',
                     'zhabei']
        districts_cn = ['浦东', '闵行', '宝山', '徐汇', '松江', '嘉定', '静安', '普陀',
                        '杨浦', '虹口', '长宁', '黄浦', '青浦', '奉贤', '金山', '崇明', '闸北']
        for dis, dis_cn in zip(districts, districts_cn):
            yield scrapy.Request(url=url + dis + "/",
                                 callback=self.parse)

    def parse(self, response):
        container = response.xpath("/html/body/div[3]/div/div[1]/dl[2]/dd/div[1]/div[2]")
        district = response.xpath("/html/body/div[3]/div/div[1]/dl[2]/dd/div[1]/div[1]").css("a.selected::text").get()
        items = container.css("a")
        with open("location.csv", mode='ab') as f:
            csv_file = csv.writer(f, encoding='utf-8')

            for item in items:
                # print(item.get())
                csv_file.writerow(
                    [district, item.xpath(".//text()").get(), "https://sh.lianjia.com" + item.xpath(".//@href").get()])
