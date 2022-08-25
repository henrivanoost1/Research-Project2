import pandas as pd
from google.cloud.vision_v1 import types
from google.cloud import vision
import io
from pytimeextractor import ExtractionService, PySettingsBuilder
import cv2
import pytesseract
import os
import sys
from pytesseract import Output
import json
# import layoutparser as lp
import re
import ast
import numpy as np
from PIL import Image
from pythonRLSA import rlsa
import phonenumbers
# from pythonRLSA import rlsa
import math
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Henri Van Oost\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('hello.png')


def read_img_vision():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\Henri Van Oost\\Documents\\MCT\\Semester5\\Research Project\\research-project-360323-4c28b9bddffc.json'

    client = vision.ImageAnnotatorClient()

    FILE_NAME = 'upload.png'
    FOLDER_PATH = 'C:/Users/Henri Van Oost/Documents/MCT/Semester5/Research Project/Henri/static/'

    with io.open(os.path.join(FOLDER_PATH, FILE_NAME), 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    df = pd.DataFrame(columns=['locale', 'description'])

    for text in texts:
        df = df.append(
            dict(locale=text.locale, description=text.description), ignore_index=True)
    output = df['description'][0]
    return output


def read_img():
    img = cv2.imread('static/upload.png')
    ConvertToText(img)
    test = ConvertToText(img)
    # print(test)
    return test


def test():
    test = "dit is een testje"
    return test


def ConvertToText(img):
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # noise removal
    # img = cv2.medianBlur(img, 5)
    # thresholding
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # dilation
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # erosion
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # opening - erosion followed by dilation
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # canny edge detection
    # img = cv2.Canny(img, 100, 200)
    # skew correction

    # -----------------------------------------------------------
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("thres.png", img)
    # -----------------------------------------------------
    text = pytesseract.image_to_string(img, lang="eng")
    os.remove("thres.png")
    text = str(text)
    # website = FindWebsite(text)
    # date = FindDate(text)
    return text


def FindTitle():
    image = cv2.imread('static/upload.png')  # reading the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert2grayscale
    (thresh, binary) = cv2.threshold(gray, 150, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
    # cv2.imshow('binary', binary)
    cv2.imwrite('binary.png', binary)

    (contours, _) = cv2.findContours(
        ~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find contours
    for contour in contours:
        """
        draw a rectangle around those contours on main image
        """
        [x, y, w, h] = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.imshow('contour', image)
    cv2.imwrite('contours.png', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # create blank image of same dimension of the original image
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    (contours, _) = cv2.findContours(
        ~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # collecting heights of each contour
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)  # average height
    # finding the larger contours
    # Applying Height heuristic
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        if h > 2*avgheight:
            cv2.drawContours(mask, [c], -1, 0, -1)
    # cv2.imshow('filter', mask)
    cv2.imwrite('filter.png', mask)
    text = pytesseract.image_to_string(
        "filter.png", lang="eng")
    # text = pytesseract.image_to_string(
    # "C:/Users/Henri Van Oost/Documents/MCT/Semester5/Research Project/Henri/filter.png", lang="eng")
    print("Titel is"+text)
    return text


def FindPhoneNumber():
    string = read_img_vision()
    regex = r"[\d]{4} [\d]{3} [\d]{4}"
    tel = re.findall(regex, string)
    print("telefon: "+str(tel))
    # tel = [x[0] for x in tel]
    tel = str(tel)
    tel = tel[2: -2].lower()
    print("telefon: "+tel)
    # print(web)
    return tel


def FindPhoneNumber2():
    string = read_img_vision()
    numbers = ''
    tel = []
    country_codes = ["AF", "AX", "AL", "DZ", "AS", "AD", "AO", "AI", "AQ", "AG", "AR",
                     "AM", "AW", "AU", "AT", "AZ", "BS", "BH", "BD", "BB", "BY", "BE",
                     "BZ", "BJ", "BM", "BT", "BO", "BQ", "BA", "BW", "BV", "BR", "IO",
                     "BN", "BG", "BF", "BI", "CV", "KH", "CM", "CA", "KY", "CF", "TD",
                     "CL", "CN", "CX", "CC", "CO", "KM", "CG", "CD", "CK", "CR", "CI",
                     "HR", "CU", "CW", "CY", "CZ", "DK", "DJ", "DM", "DO", "EC", "EG",
                     "SV", "GQ", "ER", "EE", "ET", "FK", "FO", "FJ", "FI", "FR", "GF",
                     "PF", "TF", "GA", "GM", "GE", "DE", "GH", "GI", "GR", "GL", "GD",
                     "GP", "GU", "GT", "GG", "GN", "GW", "GY", "HT", "HM", "VA", "HN",
                     "HK", "HU", "IS", "IN", "ID", "IR", "IQ", "IE", "IM", "IL", "IT",
                     "JM", "JP", "JE", "JO", "KZ", "KE", "KI", "KP", "KR", "KW", "KG",
                     "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MO", "MK",
                     "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MQ", "MR", "MU", "YT",
                     "MX", "FM", "MD", "MC", "MN", "ME", "MS", "MA", "MZ", "MM", "NA",
                     "NR", "NP", "NL", "NC", "NZ", "NI", "NE", "NG", "NU", "NF", "MP",
                     "NO", "OM", "PK", "PW", "PS", "PA", "PG", "PY", "PE", "PH", "PN",
                     "PL", "PT", "PR", "QA", "RE", "RO", "RU", "RW", "BL", "SH", "KN",
                     "LC", "MF", "PM", "VC", "WS", "SM", "ST", "SA", "SN", "RS", "SC",
                     "SL", "SG", "SX", "SK", "SI", "SB", "SO", "ZA", "GS", "SS", "ES",
                     "LK", "SD", "SR", "SJ", "SZ", "SE", "CH", "SY", "TW", "TJ", "TZ",
                     "TH", "TL", "TG", "TK", "TO", "TT", "TN", "TR", "TM", "TC", "TV",
                     "UG", "UA", "AE", "GB", "US", "UM", "UY", "UZ", "VU", "VE", "VN",
                     "VG", "VI", "WF", "EH", "YE", "ZM", "ZW"]
    # for i in country_codes:
    numbers = phonenumbers.PhoneNumberMatcher(string)
    # print(i+': '+str(numbers))
    print("hehe"+str(numbers))

    for number in numbers:
        tel.append(number)
    return numbers


def FindWebsite():
    string = read_img_vision()
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    web = [x[0] for x in url]
    web = str(web)
    web = web[2: -2].lower()
    # print(web)
    return web


# website = FindWebsite(text)
# print(website)


# text2 = "from winter to summer"
# model = lp.Detectron2LayoutModel(
#     'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config')
# layout = model.detect(img)

# text2 = "Catch the post-impressionist exhibit after 19pm! Free organ show every Sunday at 4!"

def FindDate():
    try:

        text = read_img_vision()

        settings = (PySettingsBuilder()
                    .addRulesGroup("DateGroup")
                    .excludeRules("timeRule")
                    .build()
                    )
        test = str(ExtractionService.extract(text, settings))
        result = ast.literal_eval(test)
        print(test)
        dct_test_str = str(result[0])

        # result2 = dct_test_str.replace("'", '"')

        dct_test_str = dct_test_str[24:-1]
        date = dct_test_str.split("'")[0]
        return date.lower()
    except:
        return ""
    # d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # imS = cv2.resize(img, (960, 540))

    # cv2.imshow('img', imS)
    # cv2.waitKey(0)

# print(test)

# print(test)


# result = json.loads(test)
# result = ast.literal_eval(test)

# print(result)
# print(test)
# print(result[0])
# dct_test_str = str(result[0])
# print(dct_test_str)
# result2 = dct_test_str.replace("'", '"')
# print(dct_test_str)
# result2 = json.loads(dct_test_str)
# print(str(result2))

# dct_test_str = dct_test_str[24:-1]
# print(dct_test_str.split("'")[0])

# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# imS = cv2.resize(img, (960, 540))

# cv2.imshow('img', imS)
# cv2.waitKey(0)
