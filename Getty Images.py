import requests
import re
import os
import gevent.monkey
from lxml import etree

# 全部
# url = 'https://www.gettyimages.com/search/2/image?family=editorial&sort=best#license'


class GetImgs:
    """For Getty Images"""

    def __init__(self):
        self.session = requests.session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.170 Safari/537.36',
        }

    def login(self):
        url = 'https://www.gettyimages.com/sign-in'
        req = self.session.get(url, headers=self.headers).text
        # print(req)
        login_html = etree.HTML(req)
        token = login_html.xpath('//form[@id="new_new_session"]//input[@name="authenticity_token"]/@value')[0]
        guid = login_html.xpath('//form[@id="new_new_session"]//input[@name="elqCustomerGUID"]/@value')[0]
        # pmd = login_html.xpath('//form[@id="new_new_session"]//input[@name="nds-pmd"]/@value')[0]

        url = 'https://www.gettyimages.com/sign-in'
        data = {
            'utf8': '✓',
            'authenticity_token': token,
            'new_session[return_url]': '',
            'new_session[username]': '240942649@qq.com',
            'new_session[password]': '17719302389.wang',
            'elqCustomerGUID': guid,
            'button': '',
            'nds-pmd': '{"jvqtrgQngn":{"oq":"1536:728:1536:824:1536:824","wfi":"flap-99142","oc":"q400qo6n8n86q525","fe":"1536k864 24","qvqgm":"-480","jxe":566493,"syi":"snyfr","si":"si,btt,zc4,jroz","sn":"sn,zcrt,btt,jni","us":"9416qr2n51196r8s","cy":"Jva32","sg":"{\"zgc\":0,\"gf\":snyfr,\"gr\":snyfr}","sp":"{\"gp\":gehr,\"ap\":gehr}","sf":"gehr","jt":"ss4n2rp5r9p7o3s0","sz":"5s2prn5p8o19449o","vce":"apvc,0,5oo9o6n6,2,1;fg,0,arj_frffvba_hfreanzr,0,arj_frffvba_cnffjbeq,0;zz,60p,1p0,0,;zzf,3r8,0,n,42 0,3p87 3148,1q17,1q26,-1757n,21002,-30ro;xx,1q4,0,arj_frffvba_hfreanzr;ss,0,arj_frffvba_hfreanzr;xq,0;so,8,arj_frffvba_hfreanzr;xx,1,0,arj_frffvba_cnffjbeq;ss,0,arj_frffvba_cnffjbeq;xq,0;so,7,arj_frffvba_cnffjbeq;zzf,205,3rn,n,523 332,306 6o7,157,158,-287,3q1r,729;zzf,44o,44o,n,42 0,42 0,7,7,-47p6,-121,-74n;zzf,3r9,3r9,n,ABC;zz,34p,4r7,qr,;zzf,9o,3r7,n,41 0,q328 697r,17n1,1765,-935r4,95046,2n3;zzf,3r9,3r9,n,3q8o 73p,6886 10on,10p8,10p8,-42284,26926,-41;zzf,3r7,3r7,n,ABC;zzf,3r9,3r9,n,ABC;zzf,3r8,3r8,n,ABC;zzf,3r7,3r7,n,ABC;zzf,2711,2711,32,ABC;gf,0,5491;zz,278,3p5,nq,;zzf,2499,2711,32,2np 5n7,2r0r 16s1,184,s38,-n69n,sssp,-7;zzf,270s,270s,32,ABC;gf,0,n2o1;zz,235q,393,qp,;zzf,3o3,2710,32,781 654,po7 5o5,o7,72n,-30q6,3o44,-3;zzf,2712,2712,32,ABC;gf,0,s0q3;zz,506,106,113,;zp,7n0,33,8o,;zz,103r,267,86,;zzf,n2o,270s,32,0 39,45o7 159,430,29rr,-s744,15nr6,-9;","hn":"Zbmvyyn/5.0 (Jvaqbjf AG 10.0; Jva64; k64) NccyrJroXvg/537.36 (XUGZY, yvxr Trpxb) Puebzr/68.0 Fnsnev/537.36","ns":""},"jg":"1.j-317751.1.2.zQcdhKGkwBwzvEHGdSl8DD,,.H0qdW3JnLi4gTL8Z02NhTHZRr_EcGeMpf-AUiTEaCt0jeovBekP31VB_In6Q7HHH9NQ5bYDjY43oD3aTV6BPKflx8C5JAuFs1SIT45-pHDshnPjpBxZdT0edq_F0GHYyoWsVL0CI_6ZIgVyaky4v9BKTECRarGqzJdO9F2vkualfIYEHhlGj_s_xWHEhM2wWWek_5cs3QNW3y-yY_l7verXDTahZAoMvSCkNIaklDG4B1EKRloR7glpKLJ0h_RBM"}',

        }
        self.session.post(url, data=data, headers=self.headers)

    def download(self, url_list, path):
        print('正在下载图片请耐心等待')
        for url in url_list:
            key = list(url.keys())[0]
            img = requests.get(url[key], headers=self.headers).content
            filename = path + '/' + str(key) + '.jpg'
            with open(filename, 'wb') as f:
                f.write(img)

    def download_getty_img(self, key, start=0, end=None, save_path='./', folder_name='default'):
        self.login()
        end = end - 1
        key = key.replace(' ', '-')
        # 设置路径并且创建主文件夹
        save_path = save_path if save_path.endswith('/') else save_path + '/'
        file_path = save_path + folder_name
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        # 判断页数
        start_page = start // 60 + 1 if start != 0 else 1

        # 请求第一页并且获取总页数
        url = 'https://www.gettyimages.com/photos/{key}?family=editorial&page={page}&phrase={key}&sort=best'.format(key=key, page=start_page)
        req = self.session.get(url, headers=self.headers).text
        html = etree.HTML(req)
        page = int(re.findall(r'lastPage = "(.+?)"', req)[0].replace(',', ''))
        imgs_list = html.xpath('//section[@class="search-results"]//img/@src')

        # 判断结束页
        end_page = end // 60 + 1 if end is not None else page

        # 遍历剩下的页数
        for p in range(start_page + 1, end_page + 1):
            print('获取图片链接中')
            url = 'https://www.gettyimages.com/photos/{key}?family=editorial&page={page}&phrase={key}&sort=best'.format(
                key=key, page=p)
            req = self.session.get(url, headers=self.headers).text
            html = etree.HTML(req)
            imgs = html.xpath('//section[@class="search-results"]//img/@src')
            imgs_list += imgs

        download_list = imgs_list[start:end + 1]
        gevent.monkey.patch_all()
        # 创建任务切割list
        spawn_list = []
        for i in range(5):
            spawn_list.append([])
        # 求模分割任务
        for i in range(len(download_list)):
            spawn_list[i % 5].append({i: download_list[i]})

        gevent_list = []
        for line in spawn_list:
            gevent_list.append(gevent.spawn(self.download, line, file_path))

        gevent.joinall(gevent_list)


img = GetImgs()
term_list = ['Soccer']

for term in term_list:
    print('Getting', term)
    img.download_getty_img(key=term, start=0, end=10000, save_path=r"Z:\Drive\AI\Project Daguerre\Stage 1 Phoebe\Datas"
                                                                   r"ets\GET-C", folder_name=term)
    print(term, 'finished')




