
# Import standard libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Function to load datasets
def load_datasets():
    raw_path = '../data/'
    customer_data = pd.read_csv(raw_path + 'olist_customers_dataset.csv')
    geolocation_data = pd.read_csv(raw_path + 'olist_geolocation_dataset.csv')
    order_data = pd.read_csv(raw_path + 'olist_orders_dataset.csv')
    order_items_data = pd.read_csv(raw_path + 'olist_order_items_dataset.csv')
    order_payments_data = pd.read_csv(raw_path + 'olist_order_payments_dataset.csv')
    order_reviews_data = pd.read_csv(raw_path + 'olist_order_reviews_dataset.csv')
    products_data = pd.read_csv(raw_path + 'olist_products_dataset.csv')
    sellers_data = pd.read_csv(raw_path + 'olist_sellers_dataset.csv')
    
    return (customer_data, geolocation_data, order_data, 
            order_items_data, order_payments_data, order_reviews_data, 
            products_data, sellers_data)

# Data cleaning and preprocessing function
def preprocess_data(order_data, order_reviews_data, products_data):
    # Fill missing values in orders data
    order_data['order_delivered_carrier_date'].fillna(method='ffill', inplace=True)
    order_data['order_delivered_customer_date'].fillna(method='ffill', inplace=True)

    # Fill missing values in order reviews
    order_reviews_data['review_comment_title'].fillna('No Title', inplace=True)
    order_reviews_data['review_comment_message'].fillna('No Comment', inplace=True)

    # Fill missing values in products
    products_data['product_category_name'].fillna('Unknown', inplace=True)
    products_data['product_name_lenght'].fillna(products_data['product_name_lenght'].median(), inplace=True)
    products_data['product_description_lenght'].fillna(products_data['product_description_lenght'].median(), inplace=True)
    products_data['product_photos_qty'].fillna(0, inplace=True)

    # Convert date columns to datetime format
    order_data['order_purchase_timestamp'] = pd.to_datetime(order_data['order_purchase_timestamp'])
    order_data['order_delivered_carrier_date'] = pd.to_datetime(order_data['order_delivered_carrier_date'])
    order_data['order_delivered_customer_date'] = pd.to_datetime(order_data['order_delivered_customer_date'])
    order_data['order_approved_at'] = pd.to_datetime(order_data['order_approved_at'])
    order_reviews_data['review_creation_date'] = pd.to_datetime(order_reviews_data['review_creation_date'])
    
    return order_data, order_reviews_data, products_data

# Function to merge datasets
def merge_datasets(order_data, order_items_data, order_payments_data, 
                   order_reviews_data, customer_data, products_data):
    # Merge datasets
    merged_df = order_data.merge(order_items_data, on='order_id', how='inner') \
                           .merge(order_payments_data, on='order_id', how='inner') \
                           .merge(order_reviews_data, on='order_id', how='inner') \
                           .merge(customer_data, on='customer_id', how='inner') \
                           .merge(products_data, on='product_id', how='inner')
    
    return merged_df

# Additional preprocessing steps
def additional_preprocessing(merged_df):
    # Remove duplicate rows
    merged_df.drop_duplicates(inplace=True)

    # Check for outliers in 'price' and 'freight_value'
    # Using IQR method for 'price'
    Q1 = merged_df['price'].quantile(0.25)
    Q3 = merged_df['price'].quantile(0.75)
    IQR = Q3 - Q1
    merged_df = merged_df[(merged_df['price'] >= (Q1 - 1.5 * IQR)) & (merged_df['price'] <= (Q3 + 1.5 * IQR))]

    # Standardize text columns
    merged_df['customer_city'] = merged_df['customer_city'].str.lower().str.strip()
    merged_df['product_category_name'] = merged_df['product_category_name'].str.lower().str.strip()

    # Feature Engineering: Calculate Delivery Time
    merged_df['delivery_time'] = (merged_df['order_delivered_customer_date'] - 
                                   merged_df['order_purchase_timestamp']).dt.days

    # Convert categorical columns to category data type
    merged_df['product_category_name'] = merged_df['product_category_name'].astype('category')

    # Text processing for reviews
    merged_df['review_comment_message'] = merged_df['review_comment_message'].apply(preprocess_text)

    return merged_df

def preprocess_text(text):
    """Normalize text by lowering case, removing punctuation, and stopwords."""
    # Remove punctuation and lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('portuguese'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Main function
def main():
    # Load datasets
    customer_data, geolocation_data, order_data, order_items_data, order_payments_data, order_reviews_data, products_data, sellers_data = load_datasets()
    
    # Preprocess datasets
    order_data, order_reviews_data, products_data = preprocess_data(order_data, order_reviews_data, products_data)

    # Merge datasets
    merged_df = merge_datasets(order_data, order_items_data, order_payments_data, order_reviews_data, customer_data, products_data)

    # Additional preprocessing
    merged_df = additional_preprocessing(merged_df)

    # Optionally, save the preprocessed and merged dataset to a new CSV file for future use
    raw_path = '../data/'
    merged_df.to_csv(raw_path + 'olist_cleaned_merged_data.csv', index=False)

    # Display the shape of the final merged dataset
    print(f"Merged dataset shape after cleaning: {merged_df.shape}")

if __name__ == "__main__":
    main()