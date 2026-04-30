# import libraries
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("all_data.csv")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# sidebar
with st.sidebar:
    st.title('E-commerce Dashboard')

    ## Filter rentang waktu
    start_date, end_date = st.date_input(
        label = 'Rentang Waktu Pemesanan',
        min_value = df['order_purchase_timestamp'].min().date(),
        max_value = df['order_purchase_timestamp'].max().date(),
        value = [df['order_purchase_timestamp'].min().date(), df['order_purchase_timestamp'].max().date()]
    )

# main page
## Filter data berdasarkan rentang waktu
main_df = df[(df['order_purchase_timestamp'].dt.date >= start_date) & (df['order_purchase_timestamp'].dt.date <= end_date)]

## header
st.header('E-Commerce Performance Analytics :sparkles:')

# Cek apakah data kosong setelah difilter
if main_df.empty:
    st.warning("Data tidak ditemukan pada rentang waktu tersebut. Silakan pilih rentang waktu lain (Saran: Tahun 2017 atau 2018).")
else:
    ## summary metrics (KPI)
    col1, col2, col3 = st.columns(3)
    with col1:
        total_revenue = main_df['price'].sum()
        st.metric("Total Revenue", value=f"R$ {total_revenue:,.0f}")

    with col2:
        total_orders = main_df['order_id'].nunique()
        st.metric("Total Orders", value = total_orders)

    with col3:
        avg_rating = main_df['review_score'].mean()
        st.metric("Average Review Score", value = round(avg_rating, 2))

    st.divider()
    
    ## tab/section 1 (Logistics): Menampilkan tren keterlambatan di Sao Paulo (Q2)
    st.subheader("Sao Paulo Logistics Analysis (January-June 2018)")
    sp_df = main_df[main_df['customer_city'] == 'sao paulo'].copy()
    
    if not sp_df.empty:
        sp_df['order_delivered_customer_date'] = pd.to_datetime(sp_df['order_delivered_customer_date'])
        sp_df['order_estimated_delivery_date'] = pd.to_datetime(sp_df['order_estimated_delivery_date'])
        sp_df['delay'] = (sp_df['order_delivered_customer_date'] - sp_df['order_estimated_delivery_date']).dt.days
        sp_df['month'] = sp_df['order_purchase_timestamp'].dt.strftime('%B')
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_delay = sp_df.groupby('month', observed=False)['delay'].mean().reindex(month_order).dropna().reset_index()

        fig_logistics, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_delay['month'], monthly_delay['delay'], marker='o', linewidth=2, color="#90CAF9")
        ax.axhline(0, color='red', linestyle='--') 
        ax.set_title("Rata-rata Keterlambatan Pengiriman per Bulan", fontsize=15)
        ax.set_ylabel("Selisih Hari (Minus = Lebih Cepat)")
        st.pyplot(fig_logistics)
    else:
        st.info("Tidak ada pesanan dari Sao Paulo pada rentang waktu ini.")

    ## tab/section 2 (products): Menampilkan kategori "Zona Bahaya" (Revenue tinggi vs Rating rendah) (Q3)
    st.subheader("Top Performers: Heavy Goods Sellers (> 7kg)")
    heavy_df = main_df[main_df['product_weight_g'] > 7000]
    
    if not heavy_df.empty:
        seller_res = heavy_df.groupby('seller_id').agg({'review_score': 'mean', 'order_id': 'nunique'}).reset_index()
        top_sellers = seller_res[seller_res['review_score'] > 4.0].sort_values(by='order_id', ascending=False).head(10)
        
        if not top_sellers.empty:
            fig_sellers, ax = plt.subplots(figsize=(10, 6))
            colors = ["#90CAF9"] + ["#D3D3D3"] * 9
            sns.barplot(x='order_id', y='seller_id', data=top_sellers, palette=colors, hue='seller_id', legend=False, ax=ax)
            ax.set_title("Top 10 Sellers untuk Produk Berat (>7kg)", fontsize=15)
            ax.set_xlabel("Jumlah Pesanan")
            ax.set_ylabel("Seller ID")
            st.pyplot(fig_sellers)
        else:
            st.info("Tidak ada seller yang memenuhi kriteria (Skor > 4.0) pada rentang waktu ini.")
    else:
        st.info("Tidak ada transaksi barang berat (>7kg) pada rentang waktu ini.")

    ## tab/section 3 (Sellers): Menampilkan top sellers untuk kategori barang berat (Q1)
    st.subheader("High Revenue with Low Satisfaction")
    product_res = main_df.groupby('product_category_name').agg({'price': 'sum', 'review_score': 'mean'}).reset_index()
    product_res.rename(columns={'price': 'total_revenue', 'review_score': 'avg_rating'}, inplace=True)
    
    if not product_res.empty:
        median_revenue = product_res['total_revenue'].median()
        warning_zone_df = product_res[(product_res['avg_rating'] < 3.0) & (product_res['total_revenue'] > median_revenue)].sort_values(by='avg_rating', ascending=True)
        
        if not warning_zone_df.empty:
            warning_zone_df['total_revenue'] = warning_zone_df['total_revenue'].map("R$ {:,.2f}".format)
            warning_zone_df['avg_rating'] = warning_zone_df['avg_rating'].round(2)
            st.dataframe(warning_zone_df)
        else:
            st.success("Bagus! Tidak ada produk berpendapatan tinggi dengan rating di bawah 3.0 pada rentang waktu ini.")
    else:
        st.info("Tidak ada data produk pada rentang waktu ini.")

st.caption('Copyright (c) Pyta Nur Chumairah 2026')