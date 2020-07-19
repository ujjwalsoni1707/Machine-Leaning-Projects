# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:03:38 2020

@author: Ujjwal Soni
"""


#########################################
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
#import matplotlib 
#matplotlib.rcParams["figure.figsize"] = (20,10)
from tkinter import *
from tkinter.ttk import Combobox
from tkinter import messagebox
from textblob import TextBlob

#Reading the data
df1=pd.read_csv("data.csv")

#dropping the unwanted columns
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
#print(df2.head())

#print(df2.isnull().sum())
#dropping the rows with null values
df3 = df2.dropna()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

#fucntion to handle inconsistent values of column sqft
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

#handling inconsistent values of column sqft
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]

#adding a new column of price per sqft
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']

#dimensionality reduction of column "location"
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats<=10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
#df5.to_csv("bhp.csv",index=False)

#Outlier Removal
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df8 = df7[df7.bath<df7.bhk+2]
df10 = df8.drop(['size','price_per_sqft'],axis='columns')

dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df12 = df11.drop('location',axis='columns')

#Build a Model Now...
X = df12.drop(['price'],axis='columns')
y = df12.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

#predicting the price
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

locations=list(df10['location'].unique())
#print(locations)
#print( predict_price('Badavala Nagar', 5500, 1, 3))



root= Tk()
root.geometry("500x400")
root.title("Real State Price Predictor using Machine Learning")
#root.iconbitmap("something.ico")
root.resizable(False,False)
#font=('times',15,'italic bold')
#print( predict_price('Badavala Nagar', 5500, 1, 3))

def tt(event=None):
    
    ans=predict_price(languages.get(),int(varname1.get()),int(varname2.get()),int(varname3.get()))
    
    print((languages.get(),int(varname1.get()),int(varname2.get()),int(varname3.get())))
    ans=str(ans)
    label6.configure(text=ans)
def main_exit():
    rr=messagebox.askyesnocancel('Notification','Kya Sahi Mein Jana Chahte Ho?',parent=root)
    if (rr):
        root.destroy()

locations=['1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout', '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar', '6th Phase JP Nagar', '7th Phase JP Nagar', '8th Phase JP Nagar', '9th Phase JP Nagar', 'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'Ambalipura', 'Ambedkar Nagar', 'Amruthahalli', 'Anandapura', 'Ananth Nagar', 'Anekal', 'Anjanapura', 'Ardendale', 'Arekere', 'Attibele', 'BEML Layout', 'BTM 2nd Stage', 'BTM Layout', 'Babusapalaya', 'Badavala Nagar', 'Balagere', 'Banashankari', 'Banashankari Stage II', 'Banashankari Stage III', 'Banashankari Stage V', 'Banashankari Stage VI', 'Banaswadi', 'Banjara Layout', 'Bannerghatta', 'Bannerghatta Road', 'Basavangudi', 'Basaveshwara Nagar', 'Battarahalli', 'Begur', 'Begur Road', 'Bellandur', 'Benson Town', 'Bharathi Nagar', 'Bhoganhalli', 'Billekahalli', 'Binny Pete', 'Bisuvanahalli', 'Bommanahalli', 'Bommasandra', 'Bommasandra Industrial Area', 'Bommenahalli', 'Brookefield', 'Budigere', 'CV Raman Nagar', 'Chamrajpet', 'Chandapura', 'Channasandra', 'Chikka Tirupathi', 'Chikkabanavar', 'Chikkalasandra', 'Choodasandra', 'Cooke Town', 'Cox Town', 'Cunningham Road', 'Dasanapura', 'Dasarahalli', 'Devanahalli', 'Devarachikkanahalli', 'Dodda Nekkundi', 'Doddaballapur', 'Doddakallasandra', 'Doddathoguru', 'Domlur', 'Dommasandra', 'EPIP Zone', 'Electronic City', 'Electronic City Phase II', 'Electronics City Phase 1', 'Frazer Town', 'GM Palaya', 'Garudachar Palya', 'Giri Nagar', 'Gollarapalya Hosahalli', 'Gottigere', 'Green Glen Layout', 'Gubbalala', 'Gunjur', 'HAL 2nd Stage', 'HBR Layout', 'HRBR Layout', 'HSR Layout', 'Haralur Road', 'Harlur', 'Hebbal', 'Hebbal Kempapura', 'Hegde Nagar', 'Hennur', 'Hennur Road', 'Hoodi', 'Horamavu Agara', 'Horamavu Banaswadi', 'Hormavu', 'Hosa Road', 'Hosakerehalli', 'Hoskote', 'Hosur Road', 'Hulimavu', 'ISRO Layout', 'ITPL', 'Iblur Village', 'Indira Nagar', 'JP Nagar', 'Jakkur', 'Jalahalli', 'Jalahalli East', 'Jigani', 'Judicial Layout', 'KR Puram', 'Kadubeesanahalli', 'Kadugodi', 'Kaggadasapura', 'Kaggalipura', 'Kaikondrahalli', 'Kalena Agrahara', 'Kalyan nagar', 'Kambipura', 'Kammanahalli', 'Kammasandra', 'Kanakapura', 'Kanakpura Road', 'Kannamangala', 'Karuna Nagar', 'Kasavanhalli', 'Kasturi Nagar', 'Kathriguppe', 'Kaval Byrasandra', 'Kenchenahalli', 'Kengeri', 'Kengeri Satellite Town', 'Kereguddadahalli', 'Kodichikkanahalli', 'Kodigehaali', 'Kodigehalli', 'Kodihalli', 'Kogilu', 'Konanakunte', 'Koramangala', 'Kothannur', 'Kothanur', 'Kudlu', 'Kudlu Gate', 'Kumaraswami Layout', 'Kundalahalli', 'LB Shastri Nagar', 'Laggere', 'Lakshminarayana Pura', 'Lingadheeranahalli', 'Magadi Road', 'Mahadevpura', 'Mahalakshmi Layout', 'Mallasandra', 'Malleshpalya', 'Malleshwaram', 'Marathahalli', 'Margondanahalli', 'Marsur', 'Mico Layout', 'Munnekollal', 'Murugeshpalya', 'Mysore Road', 'NGR Layout', 'NRI Layout', 'Nagarbhavi', 'Nagasandra', 'Nagavara', 'Nagavarapalya', 'Narayanapura', 'Neeladri Nagar', 'Nehru Nagar', 'OMBR Layout', 'Old Airport Road', 'Old Madras Road', 'Padmanabhanagar', 'Pai Layout', 'Panathur', 'Parappana Agrahara', 'Pattandur Agrahara', 'Poorna Pragna Layout', 'Prithvi Layout', 'R.T. Nagar', 'Rachenahalli', 'Raja Rajeshwari Nagar', 'Rajaji Nagar', 'Rajiv Nagar', 'Ramagondanahalli', 'Ramamurthy Nagar', 'Rayasandra', 'Sahakara Nagar', 'Sanjay nagar', 'Sarakki Nagar', 'Sarjapur', 'Sarjapur  Road', 'Sarjapura - Attibele Road', 'Sector 2 HSR Layout', 'Sector 7 HSR Layout', 'Seegehalli', 'Shampura', 'Shivaji Nagar', 'Singasandra', 'Somasundara Palya', 'Sompura', 'Sonnenahalli', 'Subramanyapura', 'Sultan Palaya', 'TC Palaya', 'Talaghattapura', 'Thanisandra', 'Thigalarapalya', 'Thubarahalli', 'Tindlu', 'Tumkur Road', 'Ulsoor', 'Uttarahalli', 'Varthur', 'Varthur Road', 'Vasanthapura', 'Vidyaranyapura', 'Vijayanagar', 'Vishveshwarya Layout', 'Vishwapriya Layout', 'Vittasandra', 'Whitefield', 'Yelachenahalli', 'Yelahanka', 'Yelahanka New Town', 'Yelenahalli', 'Yeshwanthpur', 'other']
#Combo box
languages= StringVar()
font_box=Combobox(root,width=30,textvariable=languages,state='readonly')
font_box['values']=locations
font_box.current(27)
font_box.place(x=170,y=50)

#Entry Box
varname1= StringVar()
entry1 = Entry(root,width=30,textvariable=varname1)
entry1.place(x=170,y=100)
varname2= StringVar()
entry2 = Entry(root,width=30,textvariable=varname2,relief='ridge')
entry2.place(x=170,y=150)
varname3= StringVar()
entry3 = Entry(root,width=30,textvariable=varname3,relief='ridge')
entry3.place(x=170,y=200)

#label
label1=Label(root,text="Plot Size in sqft :")
label1.place(x=15,y=100)
label2=Label(root,text="Number of bathrooms :")
label2.place(x=15,y=150)
label3=Label(root,text="BHK :")
label3.place(x=15,y=200)
label4=Label(root,text="Select Location :")
label4.place(x=15,y=50)
label5=Label(root,text="Predicted Price in Lakhs :")
label5.place(x=15,y=250)
label6=Label(root,text="")
label6.place(x=200,y=250)
label7=Label(root,text="Real State Price Predictor of Different Locations form Bengalore using Machine Learning")
label7.place(x=15,y=15)

#Button image
#imgbt1=PhotoImage(file="something.png")
#imgbt1=imgbt1.subample(2,2)
#fonr ke bajumein image=imgbt1,compund=RIGHT

#Button
btn1=Button(root,text="Predict",width=10,command=tt)
btn1.place(x=70,y=300)
btn2=Button(root,text="Exit",width=10,command=main_exit)
btn2.place(x=200,y=300)
root.bind('<Return>',tt)


root.mainloop()














