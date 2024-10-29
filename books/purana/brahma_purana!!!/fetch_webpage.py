from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup

options = Options()
options.binary_location = r'C:\Program Files\Mozilla Firefox\firefox.exe'
options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36")

service = Service(executable_path=r'C:\Program Files\Mozilla Firefox\geckodriver.exe')
driver = webdriver.Firefox(service=service, options=options)

with open('1-30.txt', 'w', encoding='utf-8') as output_file:
    for i in range(1, 31):
        url = f'https://www.sanskritam.world/puranas/brahmapuranam/{i}'
        driver.get(url)
        time.sleep(2)
        output_file.write(BeautifulSoup(driver.page_source, 'html.parser').get_text())
        output_file.write('\n\n\n')

with open('31-60.txt', 'w', encoding='utf-8') as output_file:
    for i in range(31, 61):
        url = f'https://www.sanskritam.world/puranas/brahmapuranam/{i}'
        driver.get(url)
        time.sleep(2)
        output_file.write(BeautifulSoup(driver.page_source, 'html.parser').get_text())
        output_file.write('\n\n\n')

driver.quit()