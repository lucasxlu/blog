---
title: "[Book] Storytelling With Data"
date: 2018-08-11 22:54:59
tags:
- Data Visualization
- Data Science
catagories:
- Data Science
- Data Visualization
- PPT
- Presentation
---
## Introduction
数据可视化(Data Visualization)是Data Science领域一个非常非常核心的内容，很多时候，我们往往会花很多力气去建模分析数据，然而最终给你的老板汇报，或者是编写分析报告的时候，通常会以图形化的方式展现。这个时候，若你能够 __利用数据讲故事__ ，那么你的汇报就会十分精彩。本文内容来自一本我个人非常喜欢的书，作者是Google工作多年、数据可视化领域的专家。若你也对数据可视化感兴趣，欢迎去阅读原著：《[Storytelling With Data](http://www.storytellingwithdata.com/book/)》

## 选择有效的图表
### 简单文本
当只有一两项数据需要分享时，简单文本是绝佳的沟通方法。考虑只用数字本身(尽可能突出)和一些辅助性文字来清晰地阐述观点。例如：

![Original Report](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/report_original.png)
> 上图用了相当多的文字和空间衬托仅仅两项数据。图表本身对数据的解读并没有多少帮助。

修改后：  
![Revised Report](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/report_revised.png)

### 表格
使用表格时需要记住一点：让设计融入背景。__让数据占据核心地位__。不要让厚重的边框和阴影与数据争夺受众的注意力。要用窄边框或者空白来区分表格的元素。
![Tables](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/tabels.png)

也可以使用 __热力图__ 辅助表格，这会使得极值更容易被观众捕捉：  
![Heatmap](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/heatmap.png)

### 需要避开的陷阱
* 不要使用饼图
* 不要使用3D效果图
* 不要使用双y轴的图，而是应该将它们分开成两个单独的图  
![Double Y-axis](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/double-y-before.png)  
![Revised Double Y-axis](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/double-y-revised.png)

## 去除杂乱
* 巧用留白
* 用对比(颜色、字号)突出要表达的元素
* 对齐使得页面更整洁
* Less is More:  
![Less is More](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/less-is-more.png)

## 聚焦观众视线
### 文字中的前注意属性
![Preattentive Attributes](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/preattentive-attributes.png)

### 图表中的前注意属性
![Without Preattentive Attributes in Graph](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/original-graph-no-attributes.png)

加入强调之后：  
![With Preattentive Attributes in Graph](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/original-graph-with-attributes.png)

### 颜色
使用少量的颜色。通常选择灰色做背景，再挑选一个大胆的颜色（例如蓝色）来吸引注意。

![Color](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/color.png)

### 避免"意大利面"式的图表策略
下图的折线图太过于杂乱，观众无法从中获取有用的信息。
![Spaghetti Graph](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/spaghetti-graph.png)

对这类图表的改进可利用 __前注意属性一次强调一根线条__。然后从空间上隔离这些线条的图表：
#### 一次只强调一根线 (颜色 + 线条粗细)
![One Line Highlighted](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/one-line-highlighted.png)

#### 空间隔离
![Vertically Apart](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/v-apart.png)

#### 混合方法 (空间上分离 + 一次只强调一根线条)
![Combined Approach](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/combined-approach-v.png)

![Combined Approach](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/combined-approach-h.png)

### 饼图的替代方案
未处理前的饼图：  
![Pie Chart](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/pie-chart.png)

#### 方案1: 直观展示数字
![Show Numbers Directly](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/numbers.png)

### 方案2: 简单条形图
![Simple Bar Graph](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/bar.png)

### 方案3: 水平堆叠条形图
![Stacked Horizontal Bar Graph](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/h-bar.png)

### 方案4: 斜率图
![Slope Graph](https://raw.githubusercontent.com/wyt930927/hexo-blog/master/source/_posts/book-storytelling-with-data/slopegraph.png)
> 通过线条的斜率很容易看出项目前后每个类别百分比的 __变化__，易于 __对比来突出项目效果__。
