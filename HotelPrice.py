import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time

def scrape_hotel_prices():
    service = Service(r"C:\Users\Asus\Downloads\chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    url = "https://www.booking.com"
    driver.get(url)
    time.sleep(3)
    search_box = driver.find_element(By.ID, "ss")
    search_box.send_keys("Bhutan")
    search_box.send_keys(Keys.ENTER)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    hotels = []
    prices = []
    for item in soup.find_all("div", class_="sr_property_block"):
        try:
            name = item.find("span", class_="sr-hotel__name").get_text(strip=True)
            price = item.find("div", class_="bui-price-display__value").get_text(strip=True)
            hotels.append(name)
            prices.append(int(price.replace("Nu", "").replace(",", "").strip()))
        except AttributeError:
            continue
    data = pd.DataFrame({"Hotel": hotels, "Price": prices})
    data.to_csv(r"C:\Users\Asus\Desktop\Data Analysis\bhutan_hotel.csv", index=False)
    return data

def analyze_and_visualize_prices():
    data = pd.read_csv(r"C:\Users\Asus\Desktop\Data Analysis\bhutan_hotel.csv")
    highest_price = data['Price'].max()
    lowest_price = data['Price'].min()
    plt.figure(figsize=(8, 5))
    plt.bar(['Highest Price', 'Lowest Price'], [highest_price, lowest_price], color=['red', 'green'])
    plt.title('Highest and Lowest Hotel Room Prices')
    plt.ylabel('Price (BTN)')
    plt.show()
    data['Days'] = np.arange(len(data))
    X = data[['Days']]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_days = pd.DataFrame({'Days': np.arange(len(data), len(data) + 10)})
    future_prices = model.predict(future_days)
    plt.figure(figsize=(10, 5))
    plt.scatter(data['Days'], data['Price'], color='blue', label='Actual Prices')
    plt.plot(data['Days'], model.predict(X), color='red', label='Regression Line')
    plt.scatter(future_days['Days'], future_prices, color='green', label='Predicted Future Prices')
    plt.xlabel("Days")
    plt.ylabel("Price (BTN)")
    plt.title("Hotel Pricing Trend and Future Prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    scrape_hotel_prices()
    analyze_and_visualize_prices()
