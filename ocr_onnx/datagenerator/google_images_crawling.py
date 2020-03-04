import urllib.request
from bs4 import BeautifulSoup
import os
import c_07
import pprint as ppr
class HexColor(object):
    def __init__(self, Purl):
        self.currentDir = os.chdir('C:/Users/dylee/PycharmProjects/ocrSecurity/ocr_onnx/datagenerator/bg_template/')
        print ("현재 작업 위치 : {}".format(os.getcwd()))

        self.url  = urllib.request.urlopen(Purl)
        self.html = self.url.read().decode('utf8')
        self.subUrl = list()
        self.colorList = list()
        self.totalColor = dict()

    def bsParsing(self):
        bsObject = BeautifulSoup(self.html, "html.parser")
        sList = bsObject.find_all("div", {"class":"colordvcon"})
        for i in sList:
            self.subUrl.append(i.a.attrs['href']) # e.g /color/5a4fcf
            self.colorList.append(i.text.strip())

    def subHtmlReq(self, url):
        indx = 1
        for u in self.subUrl:
            t_u = urllib.request.urlopen(url + u)
            t_h = t_u.read().decode('utf8')
            t_bsObject = BeautifulSoup(t_h, "html.parser")
            div_list = t_bsObject.find_all("div", {"class":"colordvconline"})

            print ("{} 번째 데이터 현재 작업 중 ...".format(indx))
            for j in self.colorList:
                t_list = list()
                for i in div_list:
                    t_list.append(i.text.strip())
                self.totalColor[j] = t_list
            print("{} 번째 데이터 현재 작업 완료...".format(indx))
            indx += 1
        #ppr.pprint (self.totalColor)

    def ColorDir(self):
        num = 1
        for k, dir_n in self.totalColor.items():
            try:
                os.mkdir(str(num) + "_" + k)
            except FileExistsError as e:
                print (e)
                pass
            else:
                # 디렉토리 이동
                os.chdir("C:/Users/dylee/PycharmProjects/ocrSecurity/ocr_onnx/datagenerator/bg_template/" + str(num) + "_" + k)
                subNumber = 1
                for i in dir_n:
                    os.mkdir(str(subNumber) + "_" + i)
                    c_07.ImageDownload(os.getcwd() + "\\" +  str(subNumber) + "_" + i, i)
                    os.chdir("C:/Users/sleep/Desktop/Total_color/" + str(num) + "_" + k)
                    subNumber += 1
                num += 1
            finally:
                os.chdir("C:/Users/dylee/PycharmProjects/ocrSecurity/ocr_onnx/datagenerator/bg_template/")


def main():
    targetURL = "http://www.color-hex.com"
    hexNode = HexColor(targetURL)
    hexNode.bsParsing()
    hexNode.subHtmlReq(targetURL)
    hexNode.ColorDir()


if __name__ == "__main__":
    main()

