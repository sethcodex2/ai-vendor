import unittest
from tools.database import Query

class TestDatabase(unittest.TestCase):
    def test_create_product(self):
        query = Query()
        response = query.add_product(name = 'Fridge', cost = '25 dollars', content = 'Second yellow product we will be using for this code. yellow. It costs 25 dollars', image = '1234324324')
        print(response)
        self.assertTrue(response == True)
    
        response = query.get_products('Yellow')
        print(response)


if __name__ == '__main__':
    unittest.main()