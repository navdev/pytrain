from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Category(Enum):
    TOOLS="tools"
    CONSUMABLES = "consumables"

class Item(BaseModel):
    name: str
    price: float
    count: int
    id: int
    category: Category

items = {
    0: Item(name="Hammer", price=9.99, count=20, id=0, category=Category.TOOLS),
    1: Item(name="Pliers", price=5.99, count=20, id=0, category=Category.TOOLS),
    2: Item(name="Nails", price=3.99, count=500, id=0, category=Category.CONSUMABLES)
}


@app.get("/")
def index() -> dict[str, dict[int, Item]]:
    return {"Items": items}