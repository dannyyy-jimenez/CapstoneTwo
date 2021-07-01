from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import base64

def search(query, path='../data/'):
    driver = webdriver.Firefox()
    driver.get(f"https://www.google.com/search?q={query}+face&tbm=isch")

    elements = driver.find_elements_by_css_selector('.rg_i.Q4LuWd')
    time.sleep(5)

    print(elements[0].get_attribute("src"))
    with open("../test-images/eminem.jpg", "wb") as fh:
        fh.write(base64.decodebytes(b'{elements[0].get_attribute("src")}'))

    driver.close()



search('Eminem')
