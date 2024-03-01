from sqlalchemy.orm import declarative_base
from sqlalchemy_searchable import make_searchable
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy_utils.types import TSVectorType
from sqlalchemy import create_engine
from sqlalchemy.orm import configure_mappers, Session
from sqlalchemy import select
from sqlalchemy_searchable import search


Base = declarative_base()
make_searchable(Base.metadata)

class Product(Base):
    __tablename__ = "product"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    content = Column(Text)
    image = Column(String(255))
    cost = Column(String(255))
    search_vector = Column(TSVectorType("name", "content"))


class Query:
    def __init__(self):
        self.engine = create_engine("postgresql://seth:seth@localhost:5434/stream")
        configure_mappers()
        Base.metadata.create_all(self.engine)


    def add_product(self, name, content, image, cost):
        session = Session(self.engine)
        product = Product(name=name, content=content, image = image, cost = cost)
        session.add(product)
        session.commit()
        return True

    
    def get_products(self, query):
        query = query.lower()
        session = Session(self.engine)
        query = search(select(Product), query)
        products = session.scalars(query).all()
        return [{'name': x.name, 'cost': x.cost} for x in products]
