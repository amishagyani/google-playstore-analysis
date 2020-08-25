# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 23:32:19 2019

@author: DELL
"""

from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import warnings 
warnings.filterwarnings("ignore") 
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt 
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
inst=[]
df = pd.read_csv("appdata1.csv")
df=df[df.Category != '1.9']
df=df.rename(index=str, columns={"Last Updated": "Year"})
df=df.rename(index=str, columns={"Android Ver": "And_ver"})
df=df.rename(index=str, columns={"Current Ver": "Cur_ver"})
df['Year'] = df.Year.astype(str)
df['Year'] = pd.to_datetime(df['Year'])
df['Year'] = pd.DatetimeIndex(df['Year']).year
apps = df["App"].values
df['Rating'] = pd.to_numeric(df['Rating'],errors='coerce')
df['Rating'] = df['Rating'].replace(np.nan, 0)
for Installs in df.select_dtypes([np.object]):
    df[Installs] = df[Installs].str.rstrip('+')
for Installs in df.select_dtypes([np.object]):
    df[Installs] = df[Installs].str.replace('Free','0')    
for Installs in df.select_dtypes([np.object]):
    df[Installs] = df[Installs].str.replace(',','')
df['Installs'] = pd.to_numeric(df['Installs'],errors='coerce')
installs=df['Installs'].values

for Type in df.select_dtypes([np.object]):
    df[Type] = df[Type].str.replace('0','Free')

#print(installs)
# check duplicates
n_duplicated = df.duplicated(subset=['App']).sum()
#print("There are {}/{} duplicated records.".format(n_duplicated, df.shape[0]))

# Check and clean type values, defer nan value processing to the next cell
df_no_dup = df.drop(df.index[df.App.duplicated()], axis=0)
print("{} records after dropping duplicated.".format(df_no_dup.shape[0]))


# check and drop NaN values
#print("NaA value statistics in each column")
print(df_no_dup.isnull().sum(axis=0),'\n')
df_no_dup = df_no_dup.dropna(subset=['Type'])
print("Column 'Type' with NaN values are dropped, {} records left.".format(df_no_dup.shape[0]))

# prepare rating dataframe

def compute_app_types(df):
    """
    Given a dataframe, compute the number 
    of free and paid apps respectively
    """
    return sum(df.Type == "Free"), sum(df.Type == 'Paid')

df_rating = df_no_dup.dropna(subset=['Rating'])
#print("There are {} free and {} paid apps in the the Rating dataframe ".format(*compute_app_types(df_rating)))

#print("Cleaned dataframe for 'Rating' has {} records.".format(df_rating.shape[0]))

df_rating=df_rating.loc[:,['Rating', 'Installs', 'Category','Type']]
#print(df_rating)


download={}
avg_download={}
percent_d={}
rat={}
avg_rat={}

c1=0
c2=0
c3=0
c4=0
c5=0

d1=[]
d2=[]

#question1
for i in df.Category.unique():
    d1=(df[(df.Category == i)].Installs).tolist() 
    if(sum(d1)>0):
        download.update({i:sum(d1)})
        avg_download.update({i:sum(d1)/len(d1)})
total=sum(download.values())
#print(total)
#max1= max(download, key=download.get)
#print("Category with highest max downloads is ") 
#print(max1, download[max1])
for key in download.keys():
    percent_d.update({key:download.get(key)/total*100})
#print(percent_d)

for i in df.Category.unique():
    d2=(df[(df.Category == i)].Rating).tolist()
    if(sum(d2)>0):
        rat.update({i:sum(d2)})
        avg_rat.update({i:sum(d2)/len(d2)})
#print(avg_rat)



##question6
#
year_install={}
for i in df.Category.unique():
    d3= (df[(df.Category == i) & (df.Year == 2016)].Installs).tolist()
    d4= (df[(df.Category == i) & (df.Year == 2017)].Installs).tolist()
    d5= (df[(df.Category == i) & (df.Year == 2018)].Installs).tolist()
    d6=d3+d4+d5
    if(sum(d6)>0):
        year_install.update({i:sum(d6)})
#total22=sum(year_install.values())
#print(total)
#print(year_install)




#print("Category with highest max downloads in the year 2016,2017 and 2018 is ") 
#print(max_year_install, year_install[max_year_install])
#
#min_year_install= min(year_install, key=year_install.get)
#print("Category with highest least downloads in the year 2016,2017 and 2018 is ") 
#print(min_year_install, year_install[min_year_install])

for i in df.Category.unique():
    d3= (df[(df.Category == i) & (df.Year == 2016)].Installs).tolist()
    d4= (df[(df.Category == i) & (df.Year == 2017)].Installs).tolist()
    d5= (df[(df.Category == i) & (df.Year == 2018)].Installs).tolist()

a16=round(sum(d3)/len(d3))
#print(a16)

a17=round(sum(d4)/len(d4))
#print(a17)

a18=round(sum(d5)/len(d5))
#print(a18)
average=[a16,a17,a18]
yearss=['2016','2017','2018']
#plt.plot(yearss,average)
#plt.show()

s1=a16+a18
diff1=a16-a18

y1=[]
v1=[]
syear=[]
vyear=[]
and_down={}
avg_and={}
for i in df.Year.unique():
    y1.append(i)
    #print(y1)
    d7= (df[(df.Year == i) & (df.And_ver == 'Varies with device')].Installs).tolist()
    
    if(sum(d7)<=0):
        and_down.update({i:sum(d7)})
        avg_and.update({i:sum(d7)})

    elif(sum(d7)>0):
        and_down.update({i:sum(d7)})
        avg_and.update({i:sum(d7)/len(d7)})
    
aa14 = avg_and.get(2014)
#print(aa14)
aa18 = avg_and.get(2018)
#print(aa18)
diff2=aa14-aa18
s2=aa14+aa18

#print(df_rating.head())


train = df[df['Category'].isin(['SPORTS', 'GAME','TRAVEL_AND_LOCAL','NEWS_AND_MAGAZINES','SOCIAL','EVENTS','ENTERTAINMENT']) ]
#print(train)

X = train[['Year','Installs']]
Y = train['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

regressor = LogisticRegression()
regressor.fit(X_train, Y_train)

predictions=regressor.predict(X_test)
#print(predictions)
predict=predictions[0]




df2=pd.read_csv("user_rev.csv")
#print(df2)
# check and drop NaN values
print("NaN value statistics in each column")
print(df2.isnull().sum(axis=0),'\n')
df2 = df2.dropna(subset=['Sentiment'])
print("Column 'Sentiment' with NaN values are dropped, {} records left.".format(df2.shape[0]))
print("NaA value statistics in each column")
print(df2.isnull().sum(axis=0),'\n')
#print(df2)
df2['Sentiment_Polarity'] = df2['Sentiment_Polarity'].abs()

#print(df2.Sentiment_Polarity)
p_rev={}
n_rev={} 
ps1=[]
ns1=[]


for i in df2.App.unique():
    ps1= (df2[(df2.App == i) & (df2.Sentiment == 'Positive')].Sentiment_Polarity).tolist()
    #print(pc1)
    if(sum(ps1)>0):
        p_rev.update({i:sum(ps1)})

for i in df2.App.unique():
    ns1= (df2[(df2.App == i) & (df2.Sentiment == 'Negative')].Sentiment_Polarity).tolist()
    #print(ns1)
    if(sum(ns1)>=0):
        n_rev.update({i:sum(ns1)})
    
#appname = StringVar(Tk())

def adjustWindow(window):
    w = 1100 # width for the window size
    h = 700 # height for the window size
    ws = screen.winfo_screenwidth() # width of the screen
    hs = screen.winfo_screenheight() # height of the screen
    x = (ws/2) - (w/2) # calculate x and y coordinates for the Tk window
    y = (hs/2) - (h/2)
    window.geometry('%dx%d+%d+%d' % (w, h, x, y)) # set the dimensions of the screen and where it is placed
    window.resizable(False, False) # disabling the resize option for the window
    window.configure(background='white')

def downloads():
    global screen2
    screen2 = Toplevel(screen1) 
    screen2.title("Download Study")
    screen2.iconbitmap('goop.ico')
    adjustWindow(screen2) # config
    screen2["bg"]="cyan4"
    
    Label(screen2, text="DOWNLOAD STUDY",font=('Helvetica 12 bold',32),fg = "white",bg="cyan4").place(x=375,y=30)

    Button(screen2, text="Percentage Download in each Category",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down1).place(x=50,y=150)
    Button(screen2, text="App Count with Specific Download Range",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down2).place(x=50,y=210)
    Button(screen2, text="Category with MAXIMUM Downloads",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down3).place(x=50,y=270)
    Button(screen2, text="Category with MINIMUM Downloads",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down4).place(x=50,y=330)
    Button(screen2, text="Category with AVG Downloads of 2,50,000+",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down5).place(x=50,y=390)
    Button(screen2, text="Number of Downloads over 206-17-18",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down6).place(x=50,y=450)
    Button(screen2, text="Predicted Category to be downloaded",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14").place(x=50,y=510)
    message8 = Label(screen2, text="" ,bg="white"  ,fg="black"  ,width=40,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message8.place(x=50, y=570)
    result8 = str(predict)
    message8.configure(text= result8) 
    
    Button(screen2, text="Category with MAX Downloads in last 3years",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down6).place(x=610,y=150)
   
    message4 = Label(screen2, text="" ,bg="white"  ,fg="black"  ,width=43,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message4.place(x=610, y=210)
    max_year_install= max(year_install, key=year_install.get)
    result4 = str(max_year_install)+" with Maximum Downloads of "+str(year_install[max_year_install])
    message4.configure(text= result4) 
    
    Button(screen2, text="Category with MIN Downloads in last 3years",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14").place(x=610,y=270)

    message5 = Label(screen2, text="" ,bg="white"  ,fg="black"  ,width=43,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message5.place(x=610, y=330)
    min_year_install= min(year_install, key=year_install.get)
    result5 = str(min_year_install)+" with Minimum Downloads of "+str(year_install[min_year_install])
    message5.configure(text= result5) 
    
    Button(screen2, text="Download Percentage in last 3years ",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down7).place(x=610,y=390)
    message6 = Label(screen2, text="" ,bg="white"  ,fg="black"  ,width=43,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message6.place(x=610, y=450)
    
    if(diff1>0):
        p1=round((diff1/s1)*100)
        result6 =" Decrease in Downloads over 206-17-18 with "+str(p1)+"%"
        message6.configure(text= result6)
    else:
        p1=-round((diff1/s1)*100)
        result6 =" Increase in Downloads over 206-17-18 with "+str(p1)+"%"
        message6.configure(text= result6)
        
    Button(screen2, text="Download Percentage without version issues ",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=down8).place(x=610,y=510)
    message7 = Label(screen2, text="" ,bg="white"  ,fg="black"  ,width=43,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message7.place(x=610, y=570)
    
    
    if(diff2>0):
        p2=round((diff2/s2)*100)
        result7 =" Decrease in Downloads over 206-17-18 with "+str(p2)+"%"
        message7.configure(text= result7)
    else:
        p2=-round((diff2/s2)*100)
        result7 =" Increase in Downloads over 206-17-18 with "+str(p2)+"%"
        message7.configure(text= result7)  
    
    Button(screen2, text="Main Page",font=('Helvetica 12 bold',16),fg = "black",bg ="gray91",command=menu).place(x=480,y=620)

def down1():
    global screen11
    screen11 = Toplevel(screen2)
    screen11.title("Percentage Download")
    screen11.iconbitmap('goop.ico')
    #adjustWindow(screen11) # config
    screen11["bg"]="cyan4"
    
    #print("\n QUESTION1 \n")
    pd_k=list(percent_d.keys())
    pd_v=list(percent_d.values())

    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
    totals=[]
    rects1 = ax.bar(pd_k, pd_v)
    ax.set_title('Percentage Download in each Category ')
    for i in ax.patches:
        totals.append(i.get_height())
    
    # set individual bar lables using above list
    total = sum(totals)
    
    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()-.03, i.get_height()+.2, \
                str(round((i.get_height()/total)*100, 2))+'%', fontsize=8,
                    color='black')
    ax.set_xticklabels(pd_k,rotation=90)
    ax.legend(['Download Percentage'])
    ax.set_xlabel('CATEGORIES')
    ax.set_ylabel('DOWNLOAD PERCENTAGE')
    canvas = FigureCanvasTkAgg(f, master=screen11)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()
        

def down2():
    global screen12
    screen12 = Toplevel(screen2) 
    screen12.title("Apps Download")
    screen12.iconbitmap('goop.ico')
    #adjustWindow(screen12) # config
    screen12["bg"]="cyan4"
    
    c1=0
    c2=0
    c3=0
    c4=0
    c5=0
    for i,j in zip(apps,installs):
        if (j>=10000 and j<50000):
            c1=c1+1
            
    for i,j in zip(apps,installs):
        if (j>=50000 and j<150000):
            c2=c2+1
        
    for i,j in zip(apps,installs):
        if (j>=150000 and j<500000):
            c3=c3+1
    #print(c3)
            
    for i,j in zip(apps,installs):
        if (j>=500000 and j<=5000000):
            c4=c4+1
    #print(c4)
            
    for i,j in zip(apps,installs):
        if (j>5000000):
            c5=c5+1

    b1_k=['10,000-50,000','50,000-150000',' 1,50,000-5,00000','5,00000-50,00000','50,00000+']
    b1_v=[c1,c2,c3,c4,c5]
    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
    
    rects1 = ax.bar(b1_k, b1_v,color=["green"])
    ax.set_title('Number of Apps having downloads')
    
    ax.legend(['Number of Apps'])
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%0.f' % float(height),
        ha='center', va='bottom')    
    ax.set_xlabel('DOWNLOADS')
    ax.set_ylabel('NUMBER OF APPS')
    canvas = FigureCanvasTkAgg(f, master=screen12)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()

def down3():
    global screen13
    screen13 = Toplevel(screen2) 
    screen13.title("Apps Download")
    screen13.iconbitmap('goop.ico')
    adjustWindow(screen13) # config
    screen13["bg"]="cyan4"
    Label(screen13, text="MAXIMUM DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="gray14").place(x=345,y=30)

    message = Label(screen13, text="" ,bg="white"  ,fg="black"  ,width=37  ,height=2, activebackground = "white" ,font=('times', 15, ' bold ')) 
    message.place(x=340, y=200)
    max_install= max(download, key=download.get) 
    result = "Category "+str(max_install)+" has "+str(download[max_install])+" downloads."
    message.configure(text= result)

def down4():
    global screen14
    screen14 = Toplevel(screen2) 
    screen14.title("Apps Download")
    screen14.iconbitmap('goop.ico')
    adjustWindow(screen14) # config
    screen14["bg"]="cyan4"
    Label(screen14, text="MINIMUM DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="gray14").place(x=345,y=30)

    message1 = Label(screen14, text="" ,bg="white"  ,fg="black"  ,width=37  ,height=2, activebackground = "white" ,font=('times', 15, ' bold ')) 
    message1.place(x=340, y=200)
    min_install= min(download, key=download.get) 
    result1 = "Category "+str(min_install)+" has "+str(download[min_install])+" downloads."
    message1.configure(text= result1)
   
    
    
def down5():
    global screen15
    screen15 = Toplevel(screen2) 
    screen15.title("Apps Download")
    screen15.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen15["bg"]="cyan4"
    #Label(screen15, text=" DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="black").place(x=265,y=30)

    avgmore={}
    for key,value in avg_download.items():
        if value >= 250000:
            ad_k=list(avg_download.keys())
            ad_v=list(avg_download.values())
    

    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
   
    rects1 = ax.bar(ad_k, ad_v,color=["yellow"])
    ax.set_title('Average Download of at least 2,50,000 ')
    
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,height,
                '%0.f' % float(height),
        ha='center', va='bottom',rotation=90) 

    ax.set_xticklabels(ad_k,rotation=90)
    ax.legend(['Average Downloads'])
    ax.set_xlabel('CATEGORIES')
    ax.set_ylabel('AVERAGE DOWNLOADS')
    canvas = FigureCanvasTkAgg(f, master=screen15)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()
  
def down6():
    global screen16
    screen16 = Toplevel(screen2) 
    screen16.title("Apps Downloaded in 2016-17-18") 
    screen16.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen16["bg"]="cyan4"
    #Label(screen15, text=" DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="black").place(x=265,y=30)

    yr_k=list(year_install.keys())
    yr_v=list(year_install.values())
    
    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
   
    rects1 = ax.bar(yr_k, yr_v,color=["darkred"])
    ax.set_title('Apps Downloaded in 2016-17-18 ',loc='right')
    
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,height,
                '%0.f' % float(height),
        ha='center', va='bottom',rotation=90) 

    ax.set_xticklabels(yr_k,rotation=90)
    ax.legend(['Apps Downloaded'])
    ax.set_xlabel('CATEGORIES')
    ax.set_ylabel('APP DOWNLOADS')
    canvas = FigureCanvasTkAgg(f, master=screen16)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()
  

def down7():
    global screen17
    screen17 = Toplevel(screen2) 
    screen17.title("Apps Downloaded in 2016-17-18") 
    screen17.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen17["bg"]="cyan4"
#    fig = plt.figure(1)
#    plt.ion()
#        
#    plt.plot(yearss,average)
#
#    canvas = FigureCanvasTkAgg(fig, master=root)
#    plot_widget = canvas.get_tk_widget()
    fig = Figure(figsize=(15,6))
    a = fig.add_subplot(111)
    a.scatter(yearss,average,color='red')
    a.plot(yearss,average,color='blue')
    a.legend(['Average Download Rate'])
    a.set_title ("Download rate over the last 3 years", fontsize=16)
    a.set_ylabel("DOWNLOADS", fontsize=14)
    a.set_xlabel("YEAR", fontsize=14)
#
    canvas = FigureCanvasTkAgg(fig, master=screen17)
    canvas.get_tk_widget().pack()
    canvas.draw()


def down8():
    global screen18
    screen18 = Toplevel(screen2) 
    screen18.title("Apps Downloaded in 2016-17-18") 
    screen18.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen18["bg"]="cyan4"
#    fig = plt.figure(1)
#    plt.ion()
#        
#    plt.plot(yearss,average)
#
#    canvas = FigureCanvasTkAgg(fig, master=root)
#    plot_widget = canvas.get_tk_widget()
    y1.sort(reverse=True)
    #print(y1) 
    
    
    v1=list(avg_and.values())
    #print(v1)
    
    fig = Figure(figsize=(15,6))
    a = fig.add_subplot(111)
    a.scatter(y1,v1,color='blue')
    a.plot(y1,v1,color='green')
    a.legend(['Average Download Rate'])
    a.set_title ("Download rate of apps without version issues", fontsize=16)
    a.set_ylabel("DOWNLOADS", fontsize=14)
    a.set_xlabel("YEAR", fontsize=14)
#
    canvas = FigureCanvasTkAgg(fig, master=screen18)
    canvas.get_tk_widget().pack()
    canvas.draw()

def ratingpage():
    global screen3
    screen3 = Toplevel(screen) 
    screen3.title("Rating Study") 
    screen3.iconbitmap('goop.ico')
    adjustWindow(screen3) # config
    screen3["bg"]="cyan4"
    
    Label(screen3, text="RATING STUDY",font=('Helvetica 12 bold',32),fg = "white",bg="cyan4").place(x=400,y=30)

    Button(screen3, text="App Category with Highest Max Ratings",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=rat_graph).place(x=370,y=150)
    message3 = Label(screen3, text="" ,bg="white"  ,fg="black"  ,width=60 ,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message3.place(x=300, y=210)
    max_ar= max(avg_rat, key=avg_rat.get) 
    result2 = str(max_ar)+" with Maximum Average Rating of "+str(avg_rat[max_ar])
    message3.configure(text= result2) 
    
    Button(screen3, text="Apps with 1,00,000+ Installs have managed to get 4.1 + Average Rating",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=rat2).place(x=265,y=270)
    Button(screen3, text="Average Rating grouped by App Type",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=rat3).place(x=380,y=330)
    Button(screen3, text="Correlation between downloads and ratings",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=rat4).place(x=355,y=390)

    Button(screen3, text="Main Page",font=('Helvetica 12 bold',16),fg = "black",bg ="gray91",command=menu).place(x=500,y=510)


def rat_graph():
    
    global screen31
    screen31 = Toplevel(screen) 
    screen31.title("Average Rating") 
    screen31.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen31["bg"]="cyan4"
    #Label(screen15, text=" DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="black").place(x=265,y=30)
    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
    ar_k=list(avg_rat.keys())
    ar_v=list(avg_rat.values())
    
    rects1 = ax.bar(ar_k, ar_v,color=["#58508d"])
    ax.set_title('Average Ratings')
    
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,height,
                '%0.2f' % float(height),
        ha='center', va='bottom') 

    ax.set_xticklabels(ar_k,rotation=90)
    ax.legend(['Average RATING'])
    ax.set_xlabel('CATEGORIES')
    ax.set_ylabel('AVERAGE RATINGS')
    canvas = FigureCanvasTkAgg(f, master=screen31)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()


def plot_target_by_group(df, target_col, group_col, figsize=(6,4), title=""):
    order = sorted(list(set(df[group_col])))
        
    stats = df.groupby(group_col).mean()[target_col]
        
    fig, ax = plt.subplots(figsize=(15,5))
        
    sns.barplot(x=group_col, y=target_col, data=df, ax=ax, order=order).set_title(title)
    ax.set(ylim=(2, 4.5))   

    fig.tight_layout()
    return stats

  
def rat2():
    
    plot_target_by_group(df_rating.query('Installs > 100000'), 'Rating','Installs', (10, 4),title= "Average Ratings for apps with 100000+ installs")


def rat3():
#    global screen35
#    screen35 = Toplevel(screen) 
#    screen35.title("Average Rating") 
#    screen35.iconbitmap('goop.ico')
#    #adjustWindow(screen15) # config
#    screen35["bg"]="cyan4"
    stats=plot_target_by_group(df_rating, 'Rating', 'Type', title="Average Rating Groupped by App Type")

    for i, s in zip(stats.index, stats):
        print("{} app has average {} {}".format(i, 'Rating',s))
    mean_rating = df_rating.Rating.mean()
    print("Mean rating: {}".format(mean_rating))
    
    

def rat4():
    sns.heatmap(df_rating.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)

def rat5():
    ds=df2.loc[:,['Sentiment_Polarity', 'Sentiment_Subjectivity']]

    sns.heatmap(ds.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)



def reviews():
    global screen4
    screen4 = Toplevel(screen) 
    screen4.title("Review Study") 
    screen4.iconbitmap('goop.ico')
    adjustWindow(screen4) # config
    screen4["bg"]="cyan4"
    
    Label(screen4, text="REVIEW STUDY",font=('Helvetica 12 bold',32),fg = "white",bg="cyan4").place(x=400,y=30)
    Button(screen4, text="Number of Positive Reviews of Apps",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=posc).place(x=90,y=150)

    message14 = Label(screen4, text="" ,bg="white"  ,fg="black"  ,width=50 ,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message14.place(x=550, y=150)
    max_pos= max(p_rev, key=p_rev.get)
    #print("App with the most positive reviews is ") 
    result14 = str(max_pos)+" App has the Highest Positive Reviews = "+str(round(p_rev[max_pos]))+"."
    message14.configure(text= result14) 
    Button(screen4, text="Positive Review Percentage",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=pospie).place(x=90,y=210)
    Button(screen4, text="Number of Negative Reviews of Apps",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=negc).place(x=90,y=270)

    message15 = Label(screen4, text="" ,bg="white"  ,fg="black"  ,width=50 ,height=2, activebackground = "white" ,font=('times', 12, ' bold ')) 
    message15.place(x=550, y=270)
    max_neg= max(n_rev, key=n_rev.get)
    #print("App with the most positive reviews is ") 
    result15 = str(max_neg)+" App has the Highest Negative Reviews = "+str(round(n_rev[max_neg]))+"."
    message15.configure(text= result15) 
    Button(screen4, text="Negative Review Percentage",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=negpie).place(x=90,y=330)
    appname = tk.StringVar(screen)
    list1 = ["10 Best Foods for You","21-Day Meditation Experience","8 Ball Pool","8fit Workouts & Meal Planner","Cut the Rope 2","Housing-Real Estate & Property","Hotwire Hotel & Car Rental App","Hotstar","Hotels Combined - Cheap deals","Hopper - Watch & Book Flights","Honkai Impact 3rd","Easy Recipes","Candy Smash","DRAGON BALL LEGENDS"] 
    droplist = OptionMenu(screen4, appname, *list1, command=lambda x: fetch_record(appname.get())) 
    appname.set('Select An App') 
    
    droplist.config(width=50) 
    droplist.grid(row=2, column=1, pady=(5,0))
    Label(screen4, text="Review Percentage of Apps",font=('Helvetica 12 bold',20),fg = "white",bg="cyan4").place(x=550,y=330)
    droplist.place(x=550,y=390)
    Button(screen4, text="Sentiment Polarity and Subjectivity",font=('Helvetica 12 bold',16),fg = "white",bg ="gray14",command=rel).place(x=90,y=390)

    Button(screen4, text="Main Page",font=('Helvetica 12 bold',16),fg = "black",bg ="gray91",command=menu).place(x=500,y=510)

def rel():
    ds=df2.loc[:,['Sentiment_Polarity', 'Sentiment_Subjectivity']]
    sns.heatmap(ds.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0)

def fetch_record(appname):
    
    datapie = df2.loc[df2['App'] == appname] #loc function is used for returning rows 
    #print(datapie)
    dp = datapie[datapie['Sentiment']=='Positive']
    dpl=len(dp)
    #print(dpl)
    data5 = datapie[datapie['Sentiment']=='Negative']
    dn=len(data5)
    #print(dn)
    data6 = datapie[datapie['Sentiment']=='Neutral']
    dnn=len(data6)
    #print(dnn)
    r_label=['Positive','Negative','Neutral']
    r_r=[dpl,dn,dnn]
    #print(r_label)
    #print(r_r)
    figureObject, axesObject = plt.subplots()
    colors1  = ("green", "red", "cyan")
    explode = (0.1, 0.1, 0.1)
    
    axesObject.pie(r_r,
    
                   explode      = explode,
    
                   colors       = colors1,
    
                   labels       = r_label,
    
                   autopct      = '%0.f',
                   shadow       = True,
                   startangle   =90)
    
     
    
    # Aspect ratio
    
    axesObject.axis('equal')
    plt.title("REVIEWS", fontdict=None, loc='left', pad=None)
    
    plt.show()


def pospie():
#    global screen41
#    screen41 = Toplevel(screen) 
#    screen41.title("Average Rating") 
#    screen41.iconbitmap('goop.ico')
#    #adjustWindow(screen15) # config
#    screen41["bg"]="cyan4"
    

    copyOfDict = dict(p_rev)
 
# Iterate over the temporary dictionary and delete corresponding key from original dictionary
    for (key, value) in copyOfDict.items() :
        if value < 40:
            del p_rev[key]
     
    p_k=list(p_rev.keys())
    #print(p_k)
    p_v=list(p_rev.values())
    #print(p_v)
    
    
    
    explode = (0.4, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3)

    colors  = ("red", "green", "orange", "cyan", "brown", "grey", "blue", "indigo", "beige", "yellow")

     
    
    # Draw the pie chart
    figureObject, axesObject = plt.subplots(figsize=(15,6))
    axesObject.pie(p_v,labels = p_k,autopct = '%.1f%%',startangle   = 90)
    
    
    # Aspect ratio
    
    axesObject.axis('equal')
    plt.title('POSITIVE REVIEWS')
    plt.show()
     
#    canvas = FigureCanvasTkAgg(figureObject, master=screen41)
#    canvas.draw()
#    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)


def negpie():
#    global screen41
#    screen41 = Toplevel(screen) 
#    screen41.title("Average Rating") 
#    screen41.iconbitmap('goop.ico')
#    #adjustWindow(screen15) # config
#    screen41["bg"]="cyan4"
    

    copyOfDict1 = dict(n_rev)
 
#Iterate over the temporary dictionary and delete corresponding key from original dictionary
    for (key, value) in copyOfDict1.items() :
        if value <14:
            del n_rev[key]
     
#print(n_rev)
#
        
    n_k=list(n_rev.keys())
    #print(p_k)
    n_v=list(n_rev.values())
    #print(p_v)
    explode = (0.4, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3)

    colors  = ("red", "green", "orange", "cyan", "brown", "grey", "blue", "indigo", "beige", "yellow")

     
    
    # Draw the pie chart
    figureObject, axesObject = plt.subplots(figsize=(15,6))
    axesObject.pie(n_v,labels = n_k,autopct = '%.1f%%',startangle   = 90)
    
    
    # Aspect ratio
    
    axesObject.axis('equal')
    plt.title('NEGATIVE REVIEWS')
    plt.show()
     


    
#
#    canvas = FigureCanvasTkAgg(figureObject, master=screen41)
#    canvas.draw()
#    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)

def posc():

    global screen41
    screen41 = Toplevel(screen) 
    screen41.title("Positive Reviews") 
    screen41.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen41["bg"]="cyan4"
    #Label(screen15, text=" DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="black").place(x=265,y=30)
    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
    copyOfDict = dict(p_rev)
 
# Iterate over the temporary dictionary and delete corresponding key from original dictionary
    for (key, value) in copyOfDict.items() :
        if value < 30:
            del p_rev[key]

    p_k=list(p_rev.keys())
    #print(p_k)
    p_v=list(p_rev.values())
    #print(p_v)
    rects1 = ax.bar(p_k, p_v,color=["green"])
    ax.set_title('POSITIVE REVIEWS')
    
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,0.5*height,
                '%0.f' % float(height),
        ha='center', va='bottom') 

    ax.set_xticklabels(p_k,rotation=90)
    ax.legend(['No. of Positive Reviews'])
    ax.set_xlabel('APPS')
    ax.set_ylabel('POSITIVE REVIEWS')
    canvas = FigureCanvasTkAgg(f, master=screen41)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()

def negc():

    global screen41
    screen42 = Toplevel(screen) 
    screen42.title("Negative Reviews") 
    screen42.iconbitmap('goop.ico')
    #adjustWindow(screen15) # config
    screen42["bg"]="cyan4"
    #Label(screen15, text=" DOWNLOAD",font=('Helvetica 12 bold',28),fg = "white",bg="black").place(x=265,y=30)
    f = Figure(figsize=(15,6), dpi=100)
    ax = f.add_subplot(111)
    copyOfDict1 = dict(n_rev)
 
#Iterate over the temporary dictionary and delete corresponding key from original dictionary
    for (key, value) in copyOfDict1.items() :
        if value <12:
            del n_rev[key]
    n_k=list(n_rev.keys())
    #print(p_k)
    n_v=list(n_rev.values())
    #print(p_v)
    rects1 = ax.bar(n_k, n_v,color=["red"])
    ax.set_title('NEGATIVE REVIEWS')
    
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,0.5*height,
                '%0.f' % float(height),
        ha='center', va='bottom') 

    ax.set_xticklabels(n_k,rotation=90)
    ax.legend(['No. of Negative Reviews'])
    ax.set_xlabel('APPS')
    ax.set_ylabel('NEGATIVE REVIEWS')
    canvas = FigureCanvasTkAgg(f, master=screen42)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH,expand=1)
    f.tight_layout()


def menu():
    global screen1
    screen1 = Toplevel(screen)
    screen1.title("MENU") 
    screen1.iconbitmap('goop.ico')
    adjustWindow(screen1)
    screen1["bg"]="cyan4"
    
    Label(screen1, text="GOOGLE  PLAYSTORE  APP  LAUNCH  STUDY",font=('Helvetica 12 bold',32),fg = "white",bg="cyan4").place(x=60,y=50)
    Button(screen1, text="DOWNLOADS",font=('Helvetica 12 bold',20),fg = "white",bg ="gray14",command=downloads).place(x=445,y=180)
    Button(screen1, text="RATINGS",font=('Helvetica 12 bold',20),fg = "white",bg ="gray14",command=ratingpage).place(x=475,y=260)
    Button(screen1, text="REVIEWS",font=('Helvetica 12 bold',20),fg = "white",bg ="gray14",command=reviews).place(x=470,y=340)
    
def main_screen(): 
    
    global screen
    screen = Tk()  # initializing the tkinter window 
    screen.title("Google PlayStore App Study")  # mentioning title of the window 
    screen.iconbitmap('goop.ico')
    adjustWindow(screen)  # configuring the window 
    
    my_image = ImageTk.PhotoImage(Image.open("b1.jpg"))
    back = Label(image=my_image)
    back.place(x=0, y=0, relwidth=1, relheight=1)
    
    lab1=Label(screen, text="GOOGLE  PLAYSTORE  APP  LAUNCH  STUDY",font=('Helvetica 12 bold',32),fg = "black",bg="gray98")
    bob1 = Button(screen, text="CLICK TO START",font=('Helvetica 12 bold',20),fg = "white",bg ="green4",command=menu)

    lab1.place(x =80,y = 100)
    bob1.place(x=425,y=400)
    screen.mainloop() 
 
main_screen()