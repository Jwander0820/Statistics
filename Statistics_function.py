# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 15:53:13 2021

@author: Jwander
"""
#統計相關函式
import numpy as np
import math
from scipy import stats  #scipy開源的Python演算法庫和數學工具包
#Example:Studnt, n=22,  2-tail
#stats.t.ppf(1-0.025, df) => 顯著性=0.05,若t-value介於其正負之間則代表沒有顯著特徵
# df = n-1 = 22-1 = 21 #自由度(degree freedom)為樣本數-1
# print (stats.t.ppf(1-0.025, 21)) = 2.079613844727662
class Student_ttest(object):
    def one_sample_ttest(self, n, mean, variance, u0):
        """ 用法:給定樣本數、樣本平均值、樣本變異數、母體平均值
        #Student t-test用於測試樣本資料與整體(母體)平均值是否有相關
        或是具有顯著差異
        :type n: int            #樣本數
        :type mean: float       #樣本平均值
        :type variance: float   #樣本變異數
        :type u0: float         #母體平均值
        :rtype: float           #計算出來的t-value
        :rtype: boolean         #是否具有顯著性，有則True無則False)
        """
        tvalue = (mean - u0) / ((variance**0.5) / (n**0.5))
        ttable = stats.t.ppf(1 - 0.025, n-1)
        if abs(tvalue) >= ttable:
            return tvalue,True
        else:
            return tvalue,False

    def one_sample_ttest_list(self, input_list, u0):
        """用法:給定樣本清單、母體平均值
        :type input_list: List[float]   #要計算t-value的清單
        :type u0: float                 #母體平均值
        :rtype: float                   #計算出來的t-value
        :rtype: boolean                 #是否具有顯著性，有則True無則False)
        """
        n = len(input_list)
        mean = sum(input_list) / n
        variance = 0
        for i in range(n):
            variance = variance + float(input_list[i] - mean)**2
        variance = variance / (n-1) #樣本變異數
        
        tvalue = (mean - u0) / ((variance**0.5) / (n**0.5))
        ttable = stats.t.ppf(1 - 0.025, n-1)
        if abs(tvalue) >= ttable:
            return tvalue,True
        else:
            return tvalue,False
    def two_sample_ttest(self, n1, n2, mean1, mean2, variance1, variance2):
        """ 用法:給定兩組樣本數、樣本平均值、樣本變異數
        #Student 2 sample t-test用於比較兩筆樣本資料的平均值是否具有顯著差異，或是拒絕虛無假設等
        :type n1,2: int            #兩組樣本其 樣本數
        :type mean1,2: float       #兩組樣本其 樣本平均值
        :type variance1,2: float   #兩組樣本其 樣本變異數
        :rtype: float              #計算出來的t-value
        :rtype: boolean            #是否具有顯著性，有則True無則False)
        """
        tvalue = (mean1 - mean2) / ((variance1/n1 + variance2/n2 )**0.5)
        ttable = stats.t.ppf(1 - 0.025, n1 + n2 - 2)
        if abs(tvalue) >= ttable:
            return tvalue,True #True代表拒絕虛無假設，具有顯著性
        else:
            return tvalue,False
    def two_sample_ttest_lsit(self, input_list1, input_list2):
        """ 用法:給定兩組樣本數其清單資料
        #Student 2 sample t-test用於比較兩筆樣本資料的平均值是否具有顯著差異，或是拒絕虛無假設等
        :type input_list1: List[float]   #要計算t-value的清單1
        :type input_list2: List[float]   #要計算t-value的清單2
        :rtype: float                    #計算出來的t-value
        :rtype: boolean                  #是否具有顯著性，有則True無則False)
        """
        n1 = len(input_list1)
        n2 = len(input_list2)
        mean1 = sum(input_list1) / n1
        mean2 = sum(input_list2) / n2
        variance1, variance2 = 0,0
        for i in range(n1):
            variance1 = variance1 + float(input_list1[i] - mean1)**2
        variance1 = variance1 / (n1-1)
        for i in range(n2):
            variance2 = variance2 + float(input_list2[i] - mean2)**2
        variance2 = variance2 / (n2-1)
        
        tvalue = (mean1 - mean2) / ((variance1/n1 + variance2/n2 )**0.5)
        ttable = stats.t.ppf(1 - 0.025, n1 + n2 - 2)
        if abs(tvalue) >= ttable:
            return tvalue,True #True代表拒絕虛無假設，具有顯著性
        else:
            return tvalue,False
    
    def pair_sample_ttest_lsit(self, input_list1, input_list2):
        """ 用法:給定兩組樣本數其清單資料
        #Student pair sample t-test用於比較兩筆樣本資料的差異平均值是否為零(清單必須成對)
        :type input_list1: List[float]   #要計算t-value的清單1
        :type input_list2: List[float]   #要計算t-value的清單2
        :rtype: float                    #計算出來的t-value
        :rtype: boolean                  #是否具有顯著性，有則True無則False)
        """
        d = []
        variance_d = 0
        for i in range(len(input_list1)):
            tmp = input_list1[i] - input_list2[i]
            d.append(tmp)
        mean_d = sum(d)/len(d)
        for i in range(len(d)):
            variance_d = variance_d + float(d[i] - mean_d)**2
        variance_d = variance_d / (len(d)-1)
        std_d = variance_d ** 0.5

        tvalue = (mean_d - 0 ) / (std_d / (len(d)**00.5))
        ttable = stats.t.ppf(1 - 0.025, len(d) - 1)
        if abs(tvalue) >= ttable:
            return tvalue,True #True代表拒絕虛無假設，具有顯著性
        else:
            return tvalue,False
        
    def two_sample_Ftest_lsit(self, input_list1, input_list2):
        """ 用法:給定兩組樣本數其清單資料
        #Student 2 sample F-test用於比較兩筆樣本資料的變異數是否有相關或是具有顯著差異，或是拒絕虛無假設等
        :type input_list1: List[float]   #要計算F-value的清單1
        :type input_list2: List[float]   #要計算F-value的清單2
        :rtype: float                    #計算出來的F-value
        :rtype: boolean                  #是否具有顯著性，有則True無則False)
        """
        n1 = len(input_list1)
        n2 = len(input_list2)
        mean1 = sum(input_list1) / n1
        mean2 = sum(input_list2) / n2
        variance1, variance2 = 0,0
        for i in range(n1):
            variance1 = variance1 + float(input_list1[i] - mean1)**2
        variance1 = variance1 / (n1-1)
        for i in range(n2):
            variance2 = variance2 + float(input_list2[i] - mean2)**2
        variance2 = variance2 / (n2-1)
        if variance1 > variance2:  #變異數大的要放分子
            Fvalue = variance1 / variance2
            Ftable = stats.f.ppf(1 - 0.025, n1 - 1, n2 - 1)
        else:
            Fvalue = variance2 / variance1
            Ftable = stats.f.ppf(1 - 0.025, n2 - 1, n1 - 1)
        if abs(Fvalue) >= Ftable:
            return Fvalue,True #True代表拒絕虛無假設，具有顯著性
        else:
            return Fvalue,False

class correlation(object):
    def Auto_correlation(self, input_list):
        """用法:給定一組樣本數據清單
        自相關，用於檢測一筆資料本身在不同時間的相關性，觀察資料在某個時候與自己一開始最接近
        :type input_list: List[float]   #要計算自相關的清單
        :rtype: List[float]             #計算出來的相關r值
        """
        n = len(input_list)
        mean = sum(input_list) / n
        down = 0
        r = []
        for i in range(n):
            tmp = (input_list[i] - mean)**2
            down = down + tmp
        for rk in range(1,n+1):
            up = 0
            for i in range(0,rk):
                up = up + (input_list[i] - mean) * (input_list[i-rk] - mean)
            x = round(up / down,5 )
            r.append(x)
        return r
    def Cross_correlation(self, input_list1, input_list2):
        """用法:給定兩組資料清單
        互相關，用於檢測兩筆資料的相關性
        :type input_list1: List[float]  #要計算互相關的清單1
        :type input_list2: List[float]  #要計算互相關的清單2
        :rtype: List[float]             #計算出來的相關r值
        :rtype: float()                 #計算出來的顯著線之值
        """
        n1 = len(input_list1)
        n2 = len(input_list2)
        if n1 != n2 :
            return False
        mean1 = sum(input_list1) / n1
        mean2 = sum(input_list2) / n2
        down = 0 ; down1 = 0 ; down2 = 0
        r = []
        for i in range(n1):
            tmp1 = (input_list1[i] - mean1)**2
            tmp2 = (input_list2[i] - mean2)**2
            down1 = down1 + tmp1
            down2 = down2 + tmp2
        down = (down1 * down2)**0.5
        for rk in range(1,n1+1):
            up = 0
            for i in range(0,rk):
                up = up + (input_list1[i] - mean1) * (input_list2[i-rk] - mean2)
            x = round(up / down,5 )
            r.append(x)
        ttable = stats.t.ppf(1 - 0.025, n1 - 2)
        signr = ( (ttable**2) / (ttable**2 + n1 - 2) )**0.5  
        # signr為顯著線之值，取正負做相關的假設，計算方法以t-table之值做計算
        # r = ( (t**2) / (t**2 + n - 2) )**0.5
        return r,signr
    
class Fourier(object):
    # 傅立葉轉換是透過分析時間序列的頻率特徵，轉換為無窮的sin和cos的總和
    # 能夠用來測試一筆資料訊號的強度，將時間軸轉換為頻率軸，觀察訊號的周期性
    def DFT(self, input_list):
        """用法:給定一組資料清單，將其作傅立葉轉換
        離散傅立葉轉換(Discrete Fourier Transform))
        :type input_list: List[float]  #要計算離散傅立葉轉換的清單
        :rtype: List[float]            #經過離散傅立葉轉換之資料
        """
        n = len(input_list)
        N = list(range(n))
        Xk_list = []
        for i in range(n):
            tc = 0
            ts = 0
            for j in range(n):
                tc = tc + input_list[j] * math.cos( (2 * math.pi * N[j] * i)/n )
                ts = ts + input_list[j] * math.sin( (2 * math.pi * N[j] * i)/n )
            tmp = tc - (-1)**0.5*ts
            Xk_list.append(tmp)
        return Xk_list
    def IFT(self, input_list):
        """用法:給定一組資料清單，可以將傅立葉轉換的值在轉換回原來的值(當然還是會略有誤差)
        反離散傅立葉轉換(Inverse Discrete Fourier Transform))
        :type input_list: List[float]  #要計算反離散傅立葉轉換的清單(經過傅立葉轉換之資料)
        :rtype: List[float]            #經過反離散傅立葉轉換之資料
        """
        n = len(input_list)
        N = list(range(n))
        IFT_list = []
        for i in range(n):
            tc = 0
            ts = 0
            for j in range(n):
                tc = tc + input_list[j] * math.cos( (2 * math.pi * N[j] * i)/n )
                ts = ts + input_list[j] * math.sin( (2 * math.pi * N[j] * i)/n )
            
            tmp = (tc / n) + (-1)**0.5*(ts / n)
            IFT_list.append(tmp)
        return IFT_list
    def One_side_Power_Spectrum(self, input_list):
        """用法:給定一組經過傅立葉轉換的資料，將其作單邊頻譜分析
        單邊頻譜分析，取傅立葉轉換後的單邊作分析(ex. n = 8，就取前 4 筆作計算)
        :type input_list: List[float]  #要計算頻譜的清單(經過傅立葉轉換之資料)
        :rtype: List[float]            #單邊頻譜分析
        """
        #頻譜分析(power spectrum)，是將傅立葉轉換完的訊號轉換成頻譜進行分析，此處為單邊(one-side)
        n = len(input_list)
        if ( n % 2 ) == 0 :
            k = int(n / 2)
        else:
            k = int(( n - 1 ) / 2)
        PS_list = []
        for i in range(k + 1):
            if i == 0:
                PS_list.append( (abs(input_list[i])**2) / n )
            else:
                PS_list.append( 2 * (abs(input_list[i])**2) / n )
        return PS_list
    def Two_side_Power_Spectrum(self, input_list):
        """用法:給定一組經過傅立葉轉換的資料，將其作雙邊頻譜分析
        雙邊頻譜分析，取傅立葉轉換後的雙邊作分析(ex. n = 8，就取前 7 筆作計算)
        :type input_list: List[float]  #要計算頻譜的清單(經過傅立葉轉換之資料)
        :rtype: List[float]            #雙邊頻譜分析
        """
        # 雙邊與單邊差異不大
        n = len(input_list)
        PS_list = []
        for i in range(n):
            PS_list.append( (abs(input_list[i])**2) / n )
        return PS_list
    def Normalized_Power_Spectrum(self, input_list_orgin, input_list):
        """用法:給定一組原始資料和傅立葉轉換的資料，將其作雙邊頻譜分析的正規化
        雙邊頻譜分析正規化，後續可以用於在白噪音和紅噪音的分析(comparing with noise)
        比較對象為白噪音的話其值等於 1 * 3，為 y = 3 的水平線
        :type input_list: List[float]  #要計算頻譜的清單(原始資料，用於計算變異數)
        :type input_list: List[float]  #要計算頻譜的清單(經過傅立葉轉換之資料)
        :rtype: List1[float]           #傅立葉轉換後的資料進行雙邊頻譜分析後的清單
        :rtype: List2[float]           #原始資料的紅噪音
        """
        n = len(input_list_orgin)
        variance = 0
        mean = sum(input_list_orgin) / n
        for i in range(n):
            variance = variance + float(input_list_orgin[i] - mean)**2
        variance = variance / (n-1) #樣本變異數
        PS_list = []  #計算雙邊頻譜分析
        for i in range(n):
            PS_list.append( ((abs(input_list[i])**2) / n) / variance )
            
        down = 0 # 計算原始資料的自相關，後續用於計算紅噪音
        r = [] 
        for i in range(n):
            tmp = (input_list_orgin[i] - mean)**2
            down = down + tmp
        for rk in range(0,n-1):
            up = 0
            for i in range(0,n-rk):
                up = up + (input_list_orgin[i] - mean) * (input_list_orgin[i+rk] - mean)
            x = round(up / down,5 )
            r.append(x) #原始資料自相關之清單
        
        a = (r[1] + ((abs(r[2]))**0.5 )) / 2  #a=(a1+sqrt(abs(a2)))/2  a1為lag-1 a2為lag-2
        Pk_list = [] #計算紅噪音
        for i in range(n):
            tmp = ( 1 - a**2 ) / (1 + a**2 - 2*a*math.cos(2*math.pi*i/n))
            Pk_list.append(tmp*3)  #噪音值為Pk*3  #大於紅噪音代表該時刻具有顯著性
        return PS_list,Pk_list

class Moving_Average(object):
    def Moving_Average(self, input_list, m):
        """用法:給定一組原始資料和移動平均的時間(m)，此處僅計算移動平均的資料
        移動平均(Moving Average)計算資料一定範圍(m)的滑動平均值
        :type input_list: List[float]  #要計算移動平均的清單(原始資料)
        :type m: int                   #要計算的滑動平均值，限制為整數
        :rtype: List[float]            #計算出來的移動平均之資料
        """
        # 注意，移動平均是抓一把資料取平均，然後往下一格資料前進，因此前後會有m/2 or (m-1)/2的資料損失
        MA = []
        m = int(m)
        n = len(input_list)
        
        for i in range(n - m + 1):
            tmp = 0
            for j in range(m):
                tmp = tmp + input_list[i + j]
            MA.append( tmp / m )
        return MA
    def Moving_Average_Align(self, input_list, m):
        """用法:給定一組原始資料和移動平均的時間(m);此處計算移動平均資料並將其與原始資料對齊!
        移動平均(Moving Average)計算資料一定範圍(m)的滑動平均值
        :type input_list: List[float]  #要計算移動平均的清單(原始資料)
        :type m: int                   #要計算的滑動平均值，限制為整數
        :rtype: List1[float]           #將原始資料去掉頭尾與移動平均對齊
        :rtype: List2[float]           #計算出來的移動平均之資料
        """
        # 注意，移動平均是抓一把資料取平均，然後往下一格資料前進，因此前後會有m/2 or (m-1)/2的資料損失
        MA = []
        m = int(m)
        n = len(input_list)
        
        for i in range(n - m + 1):
            tmp = 0
            for j in range(m):
                tmp = tmp + input_list[i + j] #m筆資料加總
            MA.append( tmp / m ) #除以m為計算平均值
        if ( m % 2 ) == 0 : #若是為偶數
            m = int( m / 2 )
            input_list = input_list[m:(n-m)]
            return input_list,MA
        else: #若是非偶數(==奇數)
            m = int( (m - 1) / 2 )
            input_list = input_list[m:(n-m)]
            return input_list,MA
        
            
                
            
        
        
        
        
        
        
        
        
        
        
        