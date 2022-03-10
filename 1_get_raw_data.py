from lxml import html
import requests
from tqdm import tqdm
import os

if not os.path.exists("data"):
    os.mkdir("data")
    
csv_data =[]
for tn in tqdm(range(1,105)):
    url1 = 'https://www.hakone-ekiden.jp/record/record02.php?tn=' + str(tn)
    page1 = requests.get(url1)
    tree1 = html.fromstring(page1.content)
    t1 = tree1.xpath('//table[@class="record layout02"]//p')
    t2 = tree1.xpath('//table[@class="record layout02"]//a')
    t3 = tree1.xpath('//table[@class="record layout02"]//a/@href')
    for i in range(int((len(t1)-8)/8)):
        url2 = 'https://www.hakone-ekiden.jp/record'+t3[i][1:]
        page2 = requests.get(url2)
        tree2 = html.fromstring(page2.content)
        t4 = tree2.xpath('//table[@class="record layout03"]//td')
        t5 = tree2.xpath('//table[@class="record layout03"]//a')
        for j in range(int(len(t5)/2)):
            line = [str(tn), str(t3[i][(t3[i].find('uid=')+4):]), 
                                 t2[i].text.replace(' ','').replace('\n',''), 
                                 t1[8+8*i+0].text,
                                 t1[8+8*i+3].text,
                                 t1[8+8*i+4].text,
                                 t1[8+8*i+5].text,
                                 t1[8+8*i+6].text,
                                 t1[8+8*i+7].text,
                                 t5[2*j].text,
                                 t5[2*j+1].text,
                                 t4[4*j+1].text,
                                 t4[4*j+3].text, url1, url2]
            line = ['' if value is None else value for value in line]
            if line[12][0:2]=='時間':
                line[12] = '0' + line[12]
            csv_data.append(','.join(line))
            csv_data.append('\n')
with open('data/1_raw_data.csv', 'w', encoding='utf-8_sig') as f:
    f.write(''.join(csv_data))
