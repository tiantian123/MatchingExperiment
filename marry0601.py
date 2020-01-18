# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:20:31 2019

@author: Chen Tian

E-mail: chentianfighting@126.com
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')
os.chdir('D:\\project\\test06')

from bokeh.plotting import figure,show,output_file
from bokeh.models import ColumnDataSource,HoverTool

'''
1、样本数据处理
** 按照一定规则生成了1万男性+1万女性样本：
** 在配对实验中，这2万个样本具有各自不同的个人属性（财富、内涵、外貌），每项属性都有一个得分
** 财富值符合指数分布，内涵和颜值符合正态分布
** 三项的平均值都为60分，标准差都为15分
'''
data_norm = pd.DataFrame({'正态分布':np.random.normal(loc = 60,scale = 15,size = 10000)})
data_exp = pd.DataFrame({'指数分布':np.random.exponential(scale=15, size=10000) + 45})
fig,axes = plt.subplots(1,2,figsize = (12,4))
data_norm.hist(bins=50,ax = axes[0],grid = True,color = 'gray',alpha = 0.6)
data_exp.hist(bins = 50,ax = axes[1],grid = True,color = 'gray',alpha = 0.6)
axes[0].grid(linestyle='--',color='gray')
axes[1].grid(linestyle='--',color='gray')
fig.savefig('01生成样本数据分布图.png')

# 创建数据函数
def CreatSample(n,name):
    sample = pd.DataFrame({'fortune':np.random.exponential(scale=15,size= n) +45,
                           'character':np.random.normal(loc=60,scale=15, size = n),
                           'apperance':np.random.normal(loc=60,scale=15, size = n)},
                           index=[name+str(i) for i in range(1,n+1)])
    sample.index.name = 'id'
    sample['score'] = sample.sum(axis=1)/3 # 综合评分
    return sample
# 生成男女样本数据
sample_m = CreatSample(10000,'m')
sample_f = CreatSample(10000,'f')
# 查看样本属性分布
fig,axes = plt.subplots(2,1,figsize = (12,8))
sample_m[['apperance','character','fortune']].iloc[:50].plot(kind='bar',colormap='Blues_r',grid = True,stacked=True,ax = axes[0],ylim = [0,300]) 
sample_f[['apperance','character','fortune']].iloc[:50].plot(kind='bar',colormap='Reds_r',grid = True,stacked=True,ax = axes[1],ylim = [0,300]) 
axes[0].grid(linestyle='--',color='gray')
axes[1].grid(linestyle='--',color='gray')
fig.savefig('02样本属性分值分布图.png')
'''
2、生成99个男性、99个女性样本数据，分别针对三种策略构建算法函数
   ** 择偶策略1：门当户对，要求双方三项指标加和的总分接近，差值不超过20分；
   ** 择偶策略2：男才女貌，男性要求女性的外貌分比自己高出至少10分，女性要求男性的财富分比自己高出至少10分；
   ** 择偶策略3：志趣相投、适度引领，要求对方的内涵得分在比自己低10分~高10分的区间内，且外貌和财富两项与自己的得分差值都在5分以内
   ** 每一轮实验中，我们将三种策略随机平分给所有样本，这里则是三种策略分别33人
   ** 这里不同策略匹配结果可能重合，所以为了简化模型 
   → 先进行策略1模拟，
   → 模拟完成后去掉该轮成功匹配的女性数据，再进行策略2模拟，
   → 模拟完成后去掉该轮成功匹配的女性数据，再进行策略3模拟
'''
sam99_m = CreatSample(99,'m')
sam99_f = CreatSample(99,'f')
sam99_m['strategy'] = np.random.choice([1,2,3],99)

# 尝试用做第一轮匹配
# 构建空的数据集，用于存储匹配成功的数据
match_success = pd.DataFrame(columns = ['m','f','round_n','strategy_type'])
# 复制样本数据
round1_f = sam99_f.copy()
round1_m = sam99_m.copy()
#给男性样本数据，随机分配策略选择 → 这里以男性为出发作为策略选择方
round1_m['choice'] = np.random.choice(round1_f.index,len(round1_m))

# 计算评判标准
round1_match = pd.merge(round1_m,round1_f,left_on='choice',right_index=True).reset_index()
round1_match['score_dis'] = np.abs(round1_match['score_x'] -round1_match['score_y'])
round1_match['app_dis'] = np.abs(round1_match['apperance_x'] -round1_match['apperance_y'])
round1_match['cha_dis'] = np.abs(round1_match['character_x'] - round1_match['character_y'])
round1_match['for_dis'] = np.abs(round1_match['fortune_x'] - round1_match['score_y'])

# 策略1：门当户对，要求双方三项指标加和的总分接近，差值不超过20分；
round1_s1_m = round1_match[round1_match['strategy'] == 1] # 筛选策略1选择的男性
# 可能多个男性选择了同一位女性，有先总分数最大的男性选择
round1_s1_sucess = round1_s1_m[round1_s1_m['score_dis']<=20].groupby('choice').max()
round1_s1_sucess = pd.merge(round1_s1_sucess,round1_m.reset_index(),left_on='score_x',right_on='score')[['id_y','choice']]
round1_s1_sucess.columns=['m','f']
round1_s1_sucess['strategy_type'] = 1
round1_s1_sucess['round_n'] = 1 # 得到策略1的成功匹配结果
round1_match.index = round1_match['choice']
round1_match = round1_match.drop(round1_s1_sucess['f'].tolist()) # 删除策略1成功被匹配的信息
# 策略2：男才女貌，男性要求女性的外貌分比自己高出至少10分，女性要求男性的财富分比自己高出至少10分；
round1_s2_m = round1_match[round1_match['strategy'] == 2] # 筛选策略2选择的男性
round1_s2_sucess = round1_s2_m[(round1_s2_m['apperance_y'] - round1_s2_m['apperance_x']>=10)&\
                               (round1_s2_m['fortune_x']-round1_s2_m['fortune_y']>=10)]
round1_s2_sucess = round1_s2_sucess.groupby('choice').max()
round1_s2_sucess = pd.merge(round1_s2_sucess,round1_m.reset_index(),left_on='score_x',right_on='score')[['id_y','choice']]
round1_s2_sucess.columns=['m','f']
round1_s2_sucess['strategy_type'] = 2
round1_s2_sucess['round_n'] = 1 # 得到策略1的成功匹配结果
round1_match.index = round1_match['choice']
round1_match = round1_match.drop(round1_s2_sucess['f'].tolist()) # 删除策略1成功被匹配的信息

# 策略3：志趣相投、适度引领，要求对方的内涵得分在比自己低10分~高10分的区间内，且外貌和财富两项与自己的得分差值都在5分以内
round1_s3_m = round1_match[round1_match['strategy'] == 3]                                          # 筛选策略3的数据 
round1_s3_success = round1_s3_m[(round1_s3_m['cha_dis'] <10) &   # 内涵得分差在10分以内
                               (round1_s3_m['for_dis'] < 5 )&    # 财富得分差在5分以内
                               (round1_s3_m['app_dis'] < 5 )]    # 外貌得分差在5分以内
round1_s3_success = round1_s3_success.groupby('choice').max()                                      # 筛选符合要求的数据
round1_s3_success = pd.merge(round1_s3_success,round1_m.reset_index(),left_on = 'score_x',right_on = 'score')[['id_y','choice']]
round1_s3_success.columns = ['m','f']
round1_s3_success['strategy_type'] = 3
round1_s3_success['round_n'] = 1    # 得到策略3的成功匹配的结果

# 筛选出成功匹配数据
match_success = pd.concat([match_success,round1_s1_sucess,round1_s2_sucess,round1_s3_success])
# 筛选出下一轮实验数据
round2_m = round1_m.drop(match_success['m'].tolist())
round2_f = round1_f.drop(match_success['f'].tolist())

def Strategy(df_m,df_f,roundnum):
    # 数据准备
    df_m['choice'] = np.random.choice(df_f.index,len(df_m))
    rnd_match = pd.merge(df_m, df_f,left_on='choice',right_index=True).reset_index()
    rnd_match['score_dis'] = np.abs(rnd_match['score_x'] - rnd_match['score_y'])
    rnd_match['for_dis'] = np.abs(rnd_match['fortune_x'] - rnd_match['fortune_y'])
    rnd_match['cha_dis'] = np.abs(rnd_match['character_x'] - rnd_match['character_y'])
    rnd_match['app_dis'] = np.abs(rnd_match['apperance_x'] - rnd_match['apperance_y'])

    # 策略1：门当户对，要求双方三项指标加和的总分接近，差值不超过20分；
    s1_m = rnd_match[rnd_match['strategy'] == 1] 
    s1_success = s1_m[s1_m['score_dis'] <= 20].groupby('choice').max()
    s1_success = pd.merge(s1_success,df_m.reset_index(),left_on = 'score_x',right_on='score')[['id_y','choice']]
    s1_success.columns = ['m','f']
    s1_success['strategy_type'] = 1
    s1_success['RoundNum'] = roundnum # 得到策略1的成功匹配的结果
    rnd_match.index = rnd_match['choice'] # 删除策略1成功匹配的女性数据
    rnd_match = rnd_match.drop(s1_success['f'].tolist())
    
    # 策略2：男才女貌，男性要求女性的外貌分比自己高出至少10分，女性要求男性的财富分比自己高出至少10分；
    s2_m = rnd_match[rnd_match['strategy'] == 2]                                          
    s2_success = s2_m[(s2_m['fortune_x'] - s2_m['fortune_y'] >= 10) & (s2_m['apperance_y'] - s2_m['apperance_x'] >= 10)] 
    s2_success = s2_success.groupby('choice').max()                               
    s2_success = pd.merge(s2_success,df_m.reset_index(),left_on = 'score_x',right_on = 'score')[['id_y','choice']]
    s2_success.columns = ['m','f']
    s2_success['strategy_type'] = 2
    s2_success['RoundNum'] = roundnum    # 得到策略2的成功匹配的结果
    rnd_match.index = rnd_match['choice']
    rnd_match = rnd_match.drop(s2_success['f'].tolist())  # 删除策略2成功匹配的女性数据

    # 策略3：志趣相投、适度引领，要求对方的内涵得分在比自己低10分~高10分的区间内，且外貌和财富两项与自己的得分差值都在5分以内
    s3_m = rnd_match[rnd_match['strategy'] == 3]                                          
    s3_success = s3_m[(s3_m['cha_dis'] <10) & (s3_m['for_dis'] < 5 ) & (s3_m['app_dis'] < 5 )]    
    s3_success = s3_success.groupby('choice').max()                                     
    s3_success = pd.merge(s3_success,df_m.reset_index(),left_on = 'score_x',right_on = 'score')[['id_y','choice']]
    s3_success.columns = ['m','f']
    s3_success['strategy_type'] = 3
    s3_success['RoundNum'] = roundnum    # 得到策略3的成功匹配的结果
    
    data_success = pd.concat([s1_success, s2_success, s3_success])
    
    return data_success

# 针对1w男性+1w女性匹配进行实验
sample_m['strategy'] = np.random.choice([1,2,3],10000)

# 复制数据
test_m1 = sample_m.copy()
test_f1 = sample_f.copy()
n = 1 # 设定实验次数变量
starttime = time.time() # 记录起始时间
# 第一轮实验测试
success_roundn = Strategy(test_m1,test_f1,n)    
match_success1 = success_roundn                                    
test_m1 = test_m1.drop(success_roundn['m'].tolist())
test_f1 = test_f1.drop(success_roundn['f'].tolist())
print('成功进行第%i轮实验，本轮实验成功匹配%i对，总共成功匹配%i对，还剩下%i位男性和%i位女性' % 
      (n,len(success_roundn),len(match_success1),len(test_m1),len(test_f1)))

# 运行模型
while len(success_roundn) !=0:
    n += 1
    success_roundn = Strategy(test_m1,test_f1,n)   
    #得到该轮成功匹配数据
    match_success1 = pd.concat([match_success1,success_roundn])           
    # 将成功匹配数据汇总
    test_m1 = test_m1.drop(success_roundn['m'].tolist())
    test_f1 = test_f1.drop(success_roundn['f'].tolist())
    # 输出下一轮实验数据
    print('成功进行第%i轮实验，本轮实验成功匹配%i对，总共成功匹配%i对，还剩下%i位男性和%i位女性' % 
          (n,len(success_roundn),len(match_success1),len(test_m1),len(test_f1)))

# 记录结束时间
endtime = time.time()

print('------------')
print('本次实验总共进行了%i轮，配对成功%i对\n------------' % (n,len(match_success1)))
print('实验总共耗时%.2f秒' % (endtime - starttime))

# ① 百分之多少的样本数据成功匹配到了对象？
print('%.2f%%的样本数据成功匹配到了对象\n---------' % (len(match_success1)/len(sample_m)*100))
# ② 采取不同择偶策略的匹配成功率分别是多少？
print('择偶策略1的匹配成功率为%.2f%%' % (len(match_success1[match_success1['strategy_type']==1])/len(sample_m[sample_m['strategy'] == 1])*100))
print('择偶策略2的匹配成功率为%.2f%%' % (len(match_success1[match_success1['strategy_type']==2])/len(sample_m[sample_m['strategy'] == 2])*100))
print('择偶策略3的匹配成功率为%.2f%%' % (len(match_success1[match_success1['strategy_type']==3])/len(sample_m[sample_m['strategy'] == 3])*100))
# ③ 采取不同择偶策略的男性各项平均分是多少？
match_m1 = pd.merge(match_success1,sample_m,left_on = 'm',right_index = True)
result_df = pd.DataFrame([{'财富均值':match_m1[match_m1['strategy_type'] == 1]['fortune'].mean(),
                          '内涵均值':match_m1[match_m1['strategy_type'] == 1]['character'].mean(),
                          '外貌均值':match_m1[match_m1['strategy_type'] == 1]['apperance'].mean()},
                         {'财富均值':match_m1[match_m1['strategy_type'] == 2]['fortune'].mean(),
                          '内涵均值':match_m1[match_m1['strategy_type'] == 2]['character'].mean(),
                          '外貌均值':match_m1[match_m1['strategy_type'] == 2]['apperance'].mean()},
                         {'财富均值':match_m1[match_m1['strategy_type'] == 3]['fortune'].mean(),
                          '内涵均值':match_m1[match_m1['strategy_type'] == 3]['character'].mean(),
                          '外貌均值':match_m1[match_m1['strategy_type'] == 3]['apperance'].mean()}],
                         index = ['择偶策略1','择偶策略2','择偶策略3'])
# 构建数据dataframe

print('择偶策略1的男性 → 财富均值为%.2f，内涵均值为%.2f，外貌均值为%.2f' % 
      (result_df.loc['择偶策略1']['财富均值'],result_df.loc['择偶策略1']['内涵均值'],result_df.loc['择偶策略1']['外貌均值']))
print('择偶策略2的男性 → 财富均值为%.2f，内涵均值为%.2f，外貌均值为%.2f' % 
      (result_df.loc['择偶策略2']['财富均值'],result_df.loc['择偶策略2']['内涵均值'],result_df.loc['择偶策略2']['外貌均值']))
print('择偶策略3的男性 → 财富均值为%.2f，内涵均值为%.2f，外貌均值为%.2f' % 
      (result_df.loc['择偶策略3']['财富均值'],result_df.loc['择偶策略3']['内涵均值'],result_df.loc['择偶策略3']['外貌均值']))
match_m1.boxplot(column = ['fortune','character','apperance'],by='strategy_type',figsize = (10,6),layout = (1,3))
plt.ylim(0,150)
plt.savefig('03三种择偶策略箱线图.png')
'''
3、以99男+99女的样本数据，绘制匹配折线图
要求：
① 生成样本数据，模拟匹配实验
② 生成绘制数据表格
③ bokhe制图
   ** 这里设置图例，并且可交互（消隐模式）
'''
# 准备数据
sam99_m = CreatSample(99,'m')
sam99_f = CreatSample(99,'f')
sam99_m['strategy'] = np.random.choice([1,2,3],99)
# 初始化数据
test_m2 = sam99_m.copy()
test_f2 = sam99_f.copy()
n = 1
starttime = time.time()
# 开始第一轮实验测试
success_roundn = Strategy(test_m2, test_f2,n)    
match_success2 = success_roundn                                      
test_m2 = test_m2.drop(success_roundn['m'].tolist())
test_f2 = test_f2.drop(success_roundn['f'].tolist())
print('成功进行第%i轮实验，本轮实验成功匹配%i对，总共成功匹配%i对，还剩下%i位男性和%i位女性' % 
      (n,len(success_roundn),len(match_success2),len(test_m2),len(test_f2)))

# 继续运行模型
while len(success_roundn) !=0:
    n += 1
    success_roundn = Strategy(test_m2,test_f2,n)   
    #得到该轮成功匹配数据
    match_success2 = pd.concat([match_success2,success_roundn])           
    # 将成功匹配数据汇总
    test_m2 = test_m2.drop(success_roundn['m'].tolist())
    test_f2 = test_f2.drop(success_roundn['f'].tolist())
    # 输出下一轮实验数据
    print('成功进行第%i轮实验，本轮实验成功匹配%i对，总共成功匹配%i对，还剩下%i位男性和%i位女性' % 
          (n,len(success_roundn),len(match_success2),len(test_m2),len(test_f2)))
# 记录结束时间
endtime = time.time()
print('**'*20)
print('本次实验总共进行了%i轮，配对成功%i对\n------------' % (n,len(match_success2)))
print('实验总共耗时%.2f秒' % (endtime - starttime))

''' 生成绘制数据表格'''
# 合并数据
graphdata1 = match_success2.copy()
graphdata1 = pd.merge(graphdata1,sam99_m,left_on = 'm',right_index = True)
graphdata1 = pd.merge(graphdata1,sam99_f,left_on = 'f',right_index = True)
# 筛选出id的数字编号，制作x，y字段
graphdata1['x'] = '0,' + graphdata1['f'].str[1:] + ',' + graphdata1['f'].str[1:]
graphdata1['x'] = graphdata1['x'].str.split(',')
graphdata1['y'] = graphdata1['m'].str[1:] + ',' + graphdata1['m'].str[1:] + ',0'
graphdata1['y'] = graphdata1['y'].str.split(',')
# 导入调色盘,设置颜色
from bokeh.palettes import brewer

round_num = graphdata1['RoundNum'].max()
color = brewer['Blues'][round_num+1]   # 这里+1是为了得到一个色带更宽的调色盘，避免最后一个颜色太浅
graphdata1['color'] = ''
for rn in graphdata1['RoundNum'].value_counts().index:
    graphdata1['color'][graphdata1['RoundNum'] == rn] = color[rn-1] 

graphdata1 = graphdata1[['m','f','strategy_type','RoundNum','score_x','score_y','x','y','color']]

# bokeh 绘图
output_file('test06_h1.html')

p1 = figure(plot_width=500, plot_height=500,title="配对实验过程模拟示意" ,tools= 'reset,wheel_zoom,pan')   # 构建绘图空间

for datai in graphdata1.values:
    p1.line(datai[-3],datai[-2],line_width=1, line_alpha = 0.8, line_color = datai[-1],line_dash = [10,4],legend= 'round %i' % datai[3])  
    p1.circle(datai[-3],datai[-2],size = 3,color = datai[-1],legend= 'round %i' % datai[3])

# 设置其他参数
p1.ygrid.grid_line_dash = [6, 4]
p1.xgrid.grid_line_dash = [6, 4]
p1.legend.location = "top_right"
p1.legend.click_policy="hide"
show(p1)
'''
4、生成“不同类型男女配对成功率”矩阵图
要求：
① 以之前1万男+1万女实验的结果为数据
② 按照财富值、内涵值、外貌值分别给三个区间，以区间来评判“男女类型”
   ** 高分（70-100分），中分（50-70分），低分（0-50分）
   ** 按照此类分布，男性女性都可以分为27中类型：财高品高颜高、财高品中颜高、财高品低颜高、... （财→财富，品→内涵，颜→外貌）
③ bokhe制图
   ** 散点图
   ** 27行*27列，散点的颜色深浅代表匹配成功率
'''
#准备数据，合并得到成功配对的男女各项分值
graphdata2 = match_success1.copy()
graphdata2 = pd.merge(graphdata2,sample_m,left_on = 'm',right_index = True)
graphdata2 = pd.merge(graphdata2,sample_f,left_on = 'f',right_index = True)
# 筛选男女的属性字段
graphdata2 = graphdata2[['m','f','apperance_x','character_x','fortune_x','apperance_y','character_y','fortune_y']]
# 指标区间划分
graphdata2['for_m'] = pd.cut(graphdata2['fortune_x'],[0,50,70,500],labels = ['财低','财中','财高'])
graphdata2['cha_m'] = pd.cut(graphdata2['character_x'],[0,50,70,500],labels = ['品低','品中','品高'])
graphdata2['app_m'] = pd.cut(graphdata2['apperance_x'],[0,50,70,500],labels = ['颜低','颜中','颜高'])
graphdata2['for_f'] = pd.cut(graphdata2['fortune_y'],[0,50,70,500],labels = ['财低','财中','财高'])
graphdata2['cha_f'] = pd.cut(graphdata2['character_y'],[0,50,70,500],labels = ['品低','品中','品高'])
graphdata2['app_f'] = pd.cut(graphdata2['apperance_y'],[0,50,70,500],labels = ['颜低','颜中','颜高'])
graphdata2['type_m'] = graphdata2['for_m'].astype(np.str) + graphdata2['cha_m'].astype(np.str) + graphdata2['app_m'].astype(np.str)
graphdata2['type_f'] = graphdata2['for_f'].astype(np.str) + graphdata2['cha_f'].astype(np.str) + graphdata2['app_f'].astype(np.str) 
# 筛选标签字段
graphdata2 = graphdata2[['m','f','type_m','type_f']]

# 匹配成功率计算
success_n = len(graphdata2)
success_chance = graphdata2.groupby(['type_m','type_f']).count().reset_index()
success_chance['chance'] = success_chance['m']/success_n
success_chance['alpha'] = (success_chance['chance'] - success_chance['chance'].min())/(success_chance['chance'].max() - success_chance['chance'].min())*8   # 设置alpha参数

# bokeh绘图
mlst = success_chance['type_m'].value_counts().index.tolist()
flst = success_chance['type_f'].value_counts().index.tolist()
source2 = ColumnDataSource(success_chance)    # 创建数据
output_file('test06_h2.html')
hover2 = HoverTool(tooltips=[("男性类别", "@type_m"),
                           ("女性类别","@type_f"),
                           ("匹配成功率","@chance")]) # 设置标签显示内容

p2 = figure(plot_width=800, plot_height=700,x_range = mlst, y_range = flst,
           title="不同类型男女配对成功率" ,x_axis_label = '男', y_axis_label = '女',    # X,Y轴label
           tools= [hover2,'reset,wheel_zoom,pan,lasso_select'])   # 构建绘图空间
# 绘制点
p2.square_cross(x = 'type_m', y = 'type_f', source = source2,size = 18 ,color = 'red',alpha = 'alpha')

# 设置其他参数
p2.ygrid.grid_line_dash = [6, 4]
p2.xgrid.grid_line_dash = [6, 4]
p2.xaxis.major_label_orientation = "vertical"
show(p2)
