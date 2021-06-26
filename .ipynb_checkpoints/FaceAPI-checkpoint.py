#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from PIL import Image,ImageDraw,ImageFont
import json


# In[ ]:


with open('secret.json') as f:
    secret_json = json.load(f)
subscription_key = secret_json['key1']


# In[ ]:


assert subscription_key


# In[ ]:


face_api_url = 'https://20210619miyazuka.cognitiveservices.azure.com/face/v1.0/detect'


# In[ ]:


image_file = r'sample_01.jpg'
with open(image_file, 'rb') as f:
    binary_img = f.read()


# In[ ]:


# ヘッダ設定
headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}
 
# パラメーターの設定
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
}
 
# POSTリクエスト
res = requests.post(face_api_url, 
                    params=params, 
                    headers=headers, 
                    #json={"url": image_url}
                    data=binary_img
                   )
 
# JSONデコード
result = res.json()


# In[ ]:


# 顔と認識された箇所に四角を描く関数
def draw_rectangle(draw, coordinates, color, width = 1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)
 


# In[ ]:


# 顔と認識された箇所に性別を描く関数
def draw_gender(draw, coordinates, text, align, font, fill):
    draw.text(coordinates, text, align = align, font = font, fill = fill)
 


# In[ ]:


# イメージオブジェクト生成
im = Image.open(image_file)
drawing = ImageDraw.Draw(im)
 
for index in range(len(result)):
    # 取得した顔情報
    image_top    = result[index]['faceRectangle']['top']
    image_left   = result[index]['faceRectangle']['left']
    image_height = result[index]['faceRectangle']['height']
    image_width  = result[index]['faceRectangle']['width']
    image_gender = result[index]['faceAttributes']['gender']
    image_age    = result[index]['faceAttributes']['age']
 
    # 関数呼び出し(四角)
    face_top_left = (image_left, image_top)
    face_bottom_right = (image_left + image_width, image_top + image_height)
    outline_width = 3
    outline_color = "Blue"
    draw_rectangle(drawing, (face_top_left, face_bottom_right), color = outline_color, width = outline_width)
 
    # 関数呼び出し(性別)
    gender_top_left = (image_left, image_top - 30)
    font = ImageFont.truetype(
        r'/System/Library/Fonts/Hiragino Sans GB.ttc',   # Mac
        #r'C:\Windows\Fonts\HGRSGU.TTC',  # Windows
        24) 
    align = 'Left'
    fill  = 'Red'
    draw_gender(drawing, gender_top_left, image_gender+":"+str(int(image_age)), align, font, fill)
 
# イメージを表示
im.show()


# In[ ]:


result


# In[ ]:




