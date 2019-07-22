from peewee import *

database = SqliteDatabase('lianjia.db')


class UnknownField(object):
    def __init__(self, *_, **__): pass


class BaseModel(Model):
    class Meta:
        database = database


class HouseDetail(BaseModel):
    available_area = TextField(null=True)
    building_area = TextField(null=True)
    building_structure = TextField(null=True)
    building_type = TextField(null=True)
    community = TextField(null=True)
    community_url = TextField(null=True)
    decoration = TextField(null=True)
    district = TextField(null=True)
    elevator_rate = TextField(null=True)
    elevator = TextField(null=True)
    face_to = TextField(null=True)
    feature = TextField(null=True)
    floor = TextField(null=True)
    house_usage = TextField(null=True)
    huxing = TextField(null=True)
    huxing_type = TextField(null=True)
    limit_year = TextField(null=True)
    location = TextField(null=True)
    mortgage = TextField(null=True)
    property_belong = TextField(null=True)
    property_year = TextField(null=True)
    subtitle = TextField(null=True)
    tags = TextField(null=True)
    title = TextField(null=True)
    total_price = TextField(null=True)
    trade_type = TextField(null=True)
    unit_price = TextField(null=True)
    url = TextField(null=True, unique=True)

    class Meta:
        table_name = 'house_detail'
        primary_key = False


class Community(BaseModel):
    district = TextField(null=True)
    location = TextField(null=True)
    average_price = TextField(null=True)
    building_count = TextField(null=True)
    building_type = TextField(null=True)
    building_year = TextField(null=True)
    community = TextField(null=True)
    huxing_count = TextField(null=True)
    url = TextField(null=True, unique=True)
    latitude = DoubleField(null=True)
    longitude = DoubleField(null=True)

    class Meta:
        table_name = 'community'
        primary_key = False
