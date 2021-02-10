import pandas as pd
import numpy as np
from plotnine import *
from pandas import DataFrame, read_csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import urllib.request
import sys
import csv
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.options import Options


import time
import glob
import os
import pathlib


def get_reviews_yelp(driver):
    comments = driver.find_elements_by_class_name('raw__373c0__3rcx7')

    comments_text = []
    for comment in comments:
        comments_text.append(comment.text)

    return comments_text[5:]


def extract_comments(driver, link, num_pages):
    driver.get(link)
    time.sleep(5)
    reviews=[]
    for i in range(num_pages):
        button = driver.find_element_by_xpath('/html/body/div[@*]/div[@*]/yelp-react-root/div/div[@*]/div/div/div[@*]/div/div[@*]/div[@*]/section[@*]/div[@*]/div/div[@*]/div[@*]/div/div[last()]/span/a')
        button.click()

        reviews_scraped = driver.find_elements_by_class_name('raw__373c0__3rcx7')
        
        #To collect all reviews in a page in a list
        for review in reviews_scraped[1:]:
            reviews.append(review.text.replace("\n", "  "))
            
        #To navigate to the next page        
        try:
            driver.find_element_by_xpath('.//a[@class="ui_button nav next primary "]').click()
        except:
            break
        
    driver.close()
    review_df = pd.DataFrame(reviews)
    return review_df


def not_nan(string):
    return (not string != string)

def get_reviews_skytrax(airlines_list,pages):
    # scrap the latest number of pages we defined
    # scrap the revies for the list of airlines defined
    # features we will scrap for each review
    pages_list = [i for i in range(1,pages+1)]
    features = ['airline','aircraft','type_travel','type_cabin', 'date','seats_rating', 'cabin_staff_rating',
                    'food_rating', 'entertainment_rating','wifi_rating','money_rating', 'review','recommended']
    
    
    dic_scraping = {}
    for feature in features :
        dic_scraping[feature] = []
    
    for airline in airlines_list:
        for page in pages_list:
            #here we collect all info we need
            req = Request('https://www.airlinequality.com/airline-reviews/'+airline+'/page/'+str(page)+'/', headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            soup = BeautifulSoup(webpage,'html.parser')

            for categories in soup.find_all('div',{'class':'body'}):
                aircraft,type_travel,type_cabin, date = np.nan, np.nan, np.nan, np.nan
                star_seats, star_cabin_staff, star_food,star_entertainment,star_wifi,star_money  = np.nan, np.nan, np.nan, np.nan, np.nan,np.nan
                recommend = 'no'
                try : 
                    review_text = str(categories.find('div'))
                    review_text = review_text[review_text.find('|')+1 : review_text.find('</div>')]
                except :
                    review_text = np.nan
                
                for ratings in categories.find_all('tr'):
                    if ratings.find('td',{'class':'review-rating-header aircraft'}) is not None :
                        aircraft = ratings.find('td',{'class':'review-value'}).getText()
                    elif ratings.find('td',{'class':'review-rating-header type_of_traveller'}) is not None :
                        type_travel = ratings.find('td',{'class':'review-value'}).getText()
                    elif ratings.find('td',{'class':'review-rating-header cabin_flown'}) is not None :
                        type_cabin = ratings.find('td',{'class':'review-value'}).getText()
                    elif ratings.find('td',{'class':'review-rating-header date_flown'}) is not None :
                        date = ratings.find('td',{'class':'review-value'}).getText()
                    elif ratings.find('td',{'class':'review-rating-header seat_comfort'}) is not None :
                        star_seats = len(ratings.find_all('span',{'class':'star fill'}))
                    elif ratings.find('td',{'class':'review-rating-header cabin_staff_service'}) is not None :
                        star_cabin_staff = len(ratings.find_all('span',{'class':'star fill'}))
                    elif ratings.find('td',{'class':'review-rating-header food_and_beverages'}) is not None :
                        star_food = len(ratings.find_all('span',{'class':'star fill'}))
                    elif ratings.find('td',{'class':'review-rating-header inflight_entertainment'}) is not None :
                        star_entertainment = len(ratings.find_all('span',{'class':'star fill'}))
                    elif ratings.find('td',{'class':'review-rating-header wifi_and_connectivity'}) is not None :
                        star_wifi = len(ratings.find_all('span',{'class':'star fill'}))
                    elif ratings.find('td',{'class':'review-rating-header value_for_money'}) is not None :
                        star_money = len(ratings.find_all('span',{'class':'star fill'}))
                    elif ratings.find('td',{'class':'review-value rating-yes'}) is not None :
                        recommend = ratings.find('td',{'class':'review-value rating-yes'}).getText()
                
            
                values = [airline,aircraft,type_travel,type_cabin, date,star_seats, star_cabin_staff,
                        star_food, star_entertainment,star_wifi,star_money, review_text,recommend]
                
                #if there is a review we add that to the dict that will be turned to a dataframe 
                if not_nan(review_text) :
                    for i in range(len(features)):
                        dic_scraping[features[i]].append(values[i])
                
    return(pd.DataFrame.from_dict(dic_scraping))


def get_reviews_tripadvisor(airline, num_pages): 
    driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
    driver.get(urls[airline])
    reviews=[]
    for i in range(num_pages):
        
        #To allow the page to load before the code runs
        time.sleep(2)
        
        #To accept cookies
        try:
            driver.find_element_by_xpath('.//button[@class="evidon-banner-acceptbutton"]').click()
        except:
            pass
        
        #To expand "Read More"
        try:
            driver.find_element_by_xpath("//span[@class='_3maEfNCR']").click()
        except:
            pass
        
        reviews_scraped = driver.find_elements_by_class_name("cPQsENeY")
        
        #To collect all reviews in a page in a list
        for review in reviews_scraped[1:]:
            reviews.append(review.text.replace("\n", "  "))
            
        #To navigate to the next page        
        try:
            driver.find_element_by_xpath('.//a[@class="ui_button nav next primary "]').click()
        except:
            break
        
    driver.close()
    review_df = pd.DataFrame(reviews)
    return review_df




if __name__ == '__main__':

    file = pathlib.Path("../Hackathon_eleven/Web Scrapping/final_reviews.csv")
    if file.exists ():
        pass
    else:
        ########################################### SKYTRAX
        file_Skytrax = pathlib.Path("../Hackathon_eleven/Web Scrapping/final_reviews.csv")

        if file_Skytrax.exists ():
            reviews_skytrax = pd.read_csv('../Hackathon_eleven/Web Scrapping/Skytrax_csv.csv', sep=',')
        else:
            req = Request('https://www.airlinequality.com/airline-reviews/air-france', headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            soup = BeautifulSoup(webpage,'html.parser')
            # We scrap for 12 most represented airlines in Europe and added'singapore-airlines','qatar-airways','emirates'
            #total of 15 airlines, 50 pages of reviews
            airlines_list = ['air-france','hop','klm-royal-dutch-airlines','transavia','lufthansa','swiss-international-air-lines','ryanair','easyjet','british-airways','iberia','vueling-airlines','level','singapore-airlines','qatar-airways','emirates']
            reviews_skytrax = get_reviews_skytrax(airlines_list,50)
            reviews_skytrax.to_csv('../Hackathon_eleven/Web Scrapping/Skytrax_csv.csv', sep='|')
        
        ########################################### TRIPADVISOR
        
        file_tripadvisor = pathlib.Path("../Hackathon_eleven/Web Scrapping/TripAdvisor_merged.csv")

        if file_tripadvisor.exists ():
            tripadvisor_dataframe = pd.read_csv("../Hackathon_eleven/Web Scrapping/TripAdvisor_merged.csv", sep='|')
        else:
            # Initializing global parameters
            # We initiliaze and close the driver separately for each airline due to changes in the DOM
            num_pages = 100
            airlines_list_tripadvisor = ["air-france","klm","ryanair","hop","emirates","lufthansa","swiss-intl-air","austrian-airlines","british-airways","virgin-atlantic-airways","aeroflot","iberia","turkish-airlines"]
            urls = {"air-france":"https://www.tripadvisor.com/Airline_Review-d8729003-Reviews-Air-France",
                    "klm":"https://www.tripadvisor.com/Airline_Review-d8729104-Reviews-KLM-Royal-Dutch-Airlines",
                    "ryanair":"https://www.tripadvisor.com/Airline_Review-d8729141-Reviews-Ryanair",
                    "hop":"https://www.tripadvisor.com/Airline_Review-d8728906-Reviews-HOP",
                    "emirates":"https://www.tripadvisor.com/Airline_Review-d8729069-Reviews-Emirates",
                    "lufthansa":"https://www.tripadvisor.com/Airline_Review-d8729113-Reviews-Lufthansa",
                    "swiss-intl-air":"https://www.tripadvisor.com/Airline_Review-d8729160-Reviews-Swiss-International-Air-Lines-SWISS",
                    "austrian-airlines":"https://www.tripadvisor.com/Airline_Review-d8729027-Reviews-Austrian-Airlines",
                    "british-airways":"https://www.tripadvisor.com/Airline_Review-d8729039-Reviews-British-Airways",
                    "virgin-atlantic-airways":"https://www.tripadvisor.com/Airline_Review-d8729182-Reviews-Virgin-Atlantic-Airways",
                    "aeroflot":"https://www.tripadvisor.com/Airline_Review-d8728987-Reviews-Aeroflot",
                    "iberia":"https://www.tripadvisor.com/Airline_Review-d8729089-Reviews-Iberia",
                    "turkish-airlines":"https://www.tripadvisor.com/Airline_Review-d8729174-Reviews-Turkish-Airlines"
                    }
            for airline in airlines_list_tripadvisor:
                df = get_reviews_tripadvisor(airline, num_pages)
                df['airline'] = airline
                df.to_csv('../Hackathon_eleven/Web Scrapping/TripAdvisorFiles/TripAdvisor_' + airline + '.csv', sep=',', index = False)
            path = r"../Hackathon_eleven/Web Scrapping/TripAdvisorFiles/"                   
            all_files_tripadvisor = glob.glob(os.path.join(path, "*.csv"))    
            list_df_tripadvisor = []
            for filename in all_files_tripadvisor:
                print(filename)
                df = pd.read_csv(filename, sep=",")
                list_df_tripadvisor.append(df)
            tripadvisor_dataframe = pd.concat(list_df_tripadvisor, axis=0, ignore_index=True)
            tripadvisor_dataframe.to_csv("../Hackathon_eleven/Web Scrapping/TripAdvisor_merged.csv", sep=',')
        
        ########################################### YELP
        file_yelp = pathlib.Path("../Hackathon_eleven/Web Scrapping/yelp_merged.csv")

        if file_yelp.exists ():
            yelp_dataframe = pd.read_csv("../Hackathon_eleven/Web Scrapping/yelp_merged.csv", sep='|')
        else:

            options = Options()
            options.headless = True
            options.add_argument("--window-size=1920,1200")

            links = {'air-france': 'https://www.yelp.com/biz/air-france-san-francisco', 
            'klm-royal-dutch-airlines': 'https://www.yelp.com/biz/klm-royal-dutch-airlines-amstelveen',
            'lufthansa': 'https://www.yelp.com/biz/lufthansa-frankfurt-am-main-3',
            'ryanair': 'https://www.yelp.com/biz/ryanair-dublin',
            'easyjet': 'https://www.yelp.com/biz/easyjet-luton',
            'british-airways': 'https://www.yelp.com/biz/british-airways-west-drayton',
            'vueling-airlines': 'https://www.yelp.com/biz/vueling-barcelona',
            'singapore-airlines': 'https://www.yelp.com/biz/singapore-airlines-san-francisco-7',
            'american-airlines': 'https://www.yelp.com/biz/american-airlines-los-angeles-5',
            'united': 'https://www.yelp.com/biz/united-airlines-los-angeles-5',
            'southwest' : 'https://www.yelp.com/biz/southwest-airlines-chicago-2'}

            airlines_list_yelp = ["air-france","klm-royal-dutch-airlines","lufthansa", "ryanair", "easyjet","british-airways","vueling-airlines"," singapore-airlines", "american-airlines","united","southwest"]
            driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
            for airline in airlines_list_yelp:
                df = extract_comments(driver, links[airline], 10)
                df['airline'] = airline
                df.to_csv('../Hackathon_eleven/Web Scrapping/YelpFiles/Yelp_' + airline + '.csv', sep=',', index = False)
            path = r"../Hackathon_eleven/Web Scrapping/YelpFiles/"                   
            all_files_yelp = glob.glob(os.path.join(path, "*.csv"))    
            print(all_files_yelp)
            list_df_yelp = []

            for filename in all_files_yelp:
                print(filename)
                df = pd.read_csv(filename, sep=",")
                list_df_yelp.append(df)
            yelp_dataframe = pd.concat(list_df_yelp, axis=0, ignore_index=True)
            yelp_dataframe.to_csv("../Hackathon_eleven/Web Scrapping/yelp_merged.csv", sep=',')

        ##### MERGING THREE DATAFRAMES

        reviews_skytrax_final = reviews_skytrax[['airline', 'review']]
        tripadvisor_dataframe_final = tripadvisor_dataframe[['airline', 'review']]
        yelp_dataframe_final = yelp_dataframe[['airline', 'review']]
        
        intermediate = tripadvisor_dataframe_final.append(reviews_skytrax_final, ignore_index = True) 
        final = intermediate.append(yelp_dataframe_final, ignore_index = True) 
        final.drop_duplicates()
        final.to_csv("../Hackathon_eleven/Web Scrapping/final_reviews.csv", sep=',')
        