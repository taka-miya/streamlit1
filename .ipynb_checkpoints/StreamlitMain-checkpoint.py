#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as dp
from PIL import Image


# In[ ]:


# タイトル表示
st.title('Streamlit 超入門')


# In[ ]:


st.write('DataFrame')
df1 = dp.DataFrame({
    "1行目":[1, 2, 3, 4],
    "2行目":[10, 200, 30, 40]
})
# 動的なテーブル
st.dataframe(df1.style.highlight_max(axis=0), width=300, height=500)
# 性的なテーブル
st.table(df1.style.highlight_max(axis=0))


# In[ ]:


# マジックコマンド マークダウン記述
'''
# 章
## 節
### 項
```
import streamlit as st
import numpy as np
import pandas as dp
```

'''


# In[ ]:


# チャート表示
# ランダムなデータを作成する
df2 = dp.DataFrame(
    np.random.rand(20, 3),
    columns=['a','b','c']
)

if st.checkbox("チャートを表示："):
    # ラインチャートを作成
    st.line_chart(df2)
    # エリアチャートを作成
    st.area_chart(df2)
    # 棒チャートを作成
    st.bar_chart(df2)


# In[ ]:


#マッププロット
df3 = dp.DataFrame(
    np.random.rand(100, 2)/[50,50] + (35.69, 139.70),
    columns=['lat','lon']
)
if st.checkbox("マップを表示："):
    st.map(df3)


# In[ ]:


# 画像を表示
st.write('Display Image')
img = Image.open('sample_01.jpg')
#st.image(img, caption='サンプル０１画像', use_column_width=True)


# In[ ]:


# インターラクティブなウィジェット
# ラジオボタン
# スライダー
# チェックボックス

# 画像表示をチェックボックスで指定する
if st.checkbox("画像を表示："):
    st.image(img, caption='サンプル０１画像', use_column_width=True)


# In[ ]:


# セレクトボックス
option = st.selectbox(
    "あなたの好きな数字を教えてください。",
    list(range(1,100))
)    


# In[ ]:


'あなたの好きな数字は', option, 'です。'


# In[ ]:


# テキスト入力
# スライダー指定
# レイアウトを整える
# サイドバーを追加する
st.sidebar.write('質問です')
text = st.sidebar.text_input('あなたの趣味は')
condition = st.sidebar.slider('あなたの体調は', 1, 100, 50)


# In[ ]:


'あなたの趣味は', text, 'です。'
'あなたの体調', condition


# In[ ]:


# ２カラムレイアウト
right_column, left_column = st.beta_columns(2)
button = left_column.button("ボタンを押すとテキストが表示")
if button:
    st.right_column.write("右カラム表示されました")


# In[ ]:




