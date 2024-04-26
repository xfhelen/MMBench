
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# 设置开始和结束字符串
start_string = "[6/8]"
end_string = "[7/8]"
relu='relu'
conv='conv'
bnorm='norm'
elewise='elementwise'
pooling='pool'
gemm='gemm'
reduce='reduce'
# other='void'
label=['relu','conv','bnorm','elewise','pooling','gemm','reduce','other']
tp=[0,0,0,0, 0, 0, 0, 0]


# 打开文件并读取内容
with open('nsys_temp_file.txt', 'r',encoding='utf-8') as file:
    # 初始化标志变量
    reading = False
    content = []

    # 逐行读取文件内容
    for line in file:
        # 检测开始字符串
        if start_string in line:
            reading = True
            continue  # 继续下一行

        # 检测结束字符串
        if end_string in line:
            reading = False
            break  # 停止读取

        # 如果标志为 True，则将行添加到内容中
        if reading:
            content.append(line)

# 显示读取到的内容

for line in content:
    #print(line.strip())  # strip() 方法用于删除行末尾的换行符
    words = line.strip().split()

    if relu in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            relu_tp = float(second_word.replace(",", ""))
            tp[0]=relu_tp+tp[0]
                # 打印结果
            #print("relu", relu_tp)


    elif conv in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            conv_tp = float(second_word.replace(",", ""))
            tp[1]=conv_tp+tp[1]
                # 打印结果
            #print("conv", conv_tp)

    elif bnorm in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            bnorm_tp = float(second_word.replace(",", ""))
            tp[2]=bnorm_tp+tp[2]
                # 打印结果
            print("bnorm", bnorm_tp)


    elif elewise in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            ele_tp = float(second_word.replace(",", ""))
            tp[3]=ele_tp+tp[3]
                # 打印结果
            #print("elewise", ele_tp)

    elif pooling in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            pool_tp = float(second_word.replace(",", ""))
            tp[4]=pool_tp+tp[4]
                # 打印结果
            #print("pooling", pool_tp)


    elif gemm in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            gemm_tp = float(second_word.replace(",", ""))
            tp[5]=gemm_tp+tp[5]
                # 打印结果
            #print("gemm", gemm_tp)

    elif reduce in line:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            reduce_tp = float(second_word.replace(",", ""))
            tp[6]=reduce_tp+tp[6]
                # 打印结果
            #print("reduce", reduce_tp)
    else:
      if len(words) >= 3:
            # 获取第二个空格与第三个空格之间的字符并转换为数字
            second_word = words[0]
            third_word = words[2]
            try:
              other_tp = float(second_word.replace(",", ""))
              tp[7]=other_tp+tp[7]
              #print("other", other_tp)

            except ValueError:
              True

# # tp
# plt.pie(tp, labels = label, autopct = '%0.4f%%',radius=1.5)
# plt.title('Proportion of Time(%)')
# plt.show()


from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.commons.utils import JsCode

total = sum(tp)
percentages = [(i / total * 100) for i in tp]

pie = (
    Pie()
    .add("tp", [(i, j) for i, j in zip(label, percentages)])
    .set_colors(
        ["blue", "green", "yellow", "red", "pink", "orange", "purple", "lilac", "pansy"]
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Proportion of Time(%)"),
        legend_opts=opts.LegendOpts(orient="vertical", pos_top="middle", pos_left="right")
        )
    .set_series_opts(
        label_opts=opts.LabelOpts(
            formatter=JsCode(
                "function(params) {return params.name + ': ' + params.value.toFixed(2) + '%';}"
            )
        )
    )
)
# pie.render_notebook()
pie.render("pie.html")
