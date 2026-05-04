# import libraries
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config untuk layout yang lebih lebar
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

# Load Dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, "all_data.csv")

# Cache data agar proses loading lebih cepat
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

df = load_data(file_path)

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
        st.metric("Total Orders", value = f"{total_orders:,}")

    with col3:
        avg_rating = main_df['review_score'].mean()
        st.metric("Average Review Score", value = round(avg_rating, 2))

    st.divider()
    
    # =====================================================================
    # SECTION 1 (Logistics Q2): Tren keterlambatan murni di Sao Paulo
    # =====================================================================
    st.subheader("Logistics Analysis: Sao Paulo Delivery Delays")
    sp_df = main_df[main_df['customer_city'] == 'sao paulo'].copy()
    
    if not sp_df.empty:
        sp_df['order_delivered_customer_date'] = pd.to_datetime(sp_df['order_delivered_customer_date'])
        sp_df['order_estimated_delivery_date'] = pd.to_datetime(sp_df['order_estimated_delivery_date'])
        
        # Perhitungan Pure Delay sesuai Notebook
        sp_df['delivery_margin_days'] = (sp_df['order_delivered_customer_date'] - sp_df['order_estimated_delivery_date']).dt.days
        sp_df['pure_delay_days'] = sp_df['delivery_margin_days'].clip(lower=0)
        sp_df['purchase_month'] = sp_df['order_purchase_timestamp'].dt.month
        
        # Aggregation
        monthly_trend = sp_df.groupby('purchase_month')[['pure_delay_days']].mean().reset_index()
        
        category_analysis = sp_df.groupby('product_category_name').agg(
            avg_delay_days=('pure_delay_days', 'mean'),
            total_order_cases=('order_id', 'nunique')
        ).reset_index()
        
        # Filter N > 30 agar tidak misleading
        top_10_filtered = category_analysis[category_analysis['total_order_cases'] >= 30] \
            .sort_values(by='avg_delay_days', ascending=False).head(10)

        # Plotting 2 visualisasi
        fig_logistics, ax = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))
        
        # Plot 1: Line Chart
        sns.lineplot(x='purchase_month', y='pure_delay_days', data=monthly_trend, 
                     marker='o', color='crimson', linewidth=2.5, markersize=8, ax=ax[0])
        ax[0].set_title('Tren Rata-rata Keterlambatan Murni di Sao Paulo', fontsize=15, pad=15, fontweight='bold')
        ax[0].set_xlabel('Bulan Pembelian', fontsize=12)
        ax[0].set_ylabel('Rata-rata Terlambat (Hari)', fontsize=12)
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Bar Chart
        if not top_10_filtered.empty:
            sns.barplot(x='avg_delay_days', y='product_category_name', data=top_10_filtered,
                        palette='Reds_r', hue='product_category_name', legend=False, ax=ax[1])
            ax[1].set_title('Top 10 Kategori Produk dengan Rata-rata Keterlambatan Terburuk (Min. 30 Pesanan)', fontsize=15, pad=15, fontweight='bold')
            ax[1].set_xlabel('Rata-rata Keterlambatan (Hari)', fontsize=12)
            ax[1].set_ylabel('Kategori Produk', fontsize=12)
            
            for p in ax[1].patches:
                ax[1].annotate(f' {p.get_width():.1f} Hari', (p.get_width(), p.get_y() + p.get_height() / 2.),
                               ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=11)
            ax[1].set_xlim(0, top_10_filtered['avg_delay_days'].max() * 1.2)
        else:
            ax[1].set_title("Tidak ada data kategori dengan jumlah pesanan >= 30", fontsize=12)

        plt.tight_layout()
        st.pyplot(fig_logistics)
    else:
        st.info("Tidak ada pesanan dari Sao Paulo pada rentang waktu ini.")

    st.divider()

    # =====================================================================
    # SECTION 2 (Sellers Q1): Top Sellers Barang Berat
    # =====================================================================
    st.subheader("Top Performers: Heavy Goods Sellers (> 7kg)")
    heavy_df = main_df[main_df['product_weight_g'] > 7000].copy()
    
    if not heavy_df.empty:
        seller_res = heavy_df.groupby('seller_id').agg(
            total_heavy_orders=('order_id', 'nunique'),
            avg_review_score=('review_score', 'mean')
        ).reset_index()
        
        top_10_sellers = seller_res[seller_res['avg_review_score'] > 4.0].sort_values(by='total_heavy_orders', ascending=False).head(10).copy()
        
        if not top_10_sellers.empty:
            # Gunakan Truncated ID dari notebook
            top_10_sellers['seller_id_short'] = top_10_sellers['seller_id'].str[:8] + "..."

            fig_sellers, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(
                x='total_heavy_orders', 
                y='seller_id_short', 
                data=top_10_sellers, 
                palette='Blues_r', 
                hue='seller_id_short', 
                legend=False, 
                ax=ax
            )
            ax.set_title("Top 10 Sellers Barang Berat (>7kg) dengan Skor > 4.0", fontsize=16, pad=20, fontweight='bold')
            ax.set_xlabel("Jumlah Pesanan yang Berhasil", fontsize=12)
            ax.set_ylabel("Seller ID (Truncated)", fontsize=12)
            
            # Merapikan visualisasi
            ax.set_xticks([])
            sns.despine(bottom=True, left=False)
            
            for i, p in enumerate(ax.patches):
                count = int(p.get_width())
                avg_score = top_10_sellers['avg_review_score'].iloc[i]
                ax.annotate(
                    f' {count} Pesanan  |  Avg ⭐: {avg_score:.2f}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', xytext=(8, 0), textcoords='offset points',
                    fontsize=11, fontweight='bold', color='#333333'
                )
            ax.set_xlim(0, top_10_sellers['total_heavy_orders'].max() * 1.3)
            
            plt.tight_layout()
            st.pyplot(fig_sellers)
        else:
            st.info("Tidak ada seller yang memenuhi kriteria (Skor > 4.0) pada rentang waktu ini.")
    else:
        st.info("Tidak ada transaksi barang berat (>7kg) pada rentang waktu ini.")

    st.divider()

    # =====================================================================
    # SECTION 3 (Products Q3): High Revenue with Low Satisfaction
    # =====================================================================
    st.subheader("Warning Zone: High Revenue but Low Satisfaction (Rating < 3.0)")
    product_res = main_df.groupby('product_category_name').agg(
        total_revenue=('price', 'sum'), 
        avg_review_score=('review_score', 'mean')
    ).reset_index()
    
    if not product_res.empty:
        # Sesuai notebook: filter review < 3.0, ambil top 5 revenue tertinggi
        problematic_cats = product_res[product_res['avg_review_score'] < 3.0].sort_values(by='total_revenue', ascending=False).head(5).copy()
        
        if not problematic_cats.empty:
            fig_q3, ax = plt.subplots(figsize=(14, 7))
            sns.barplot(
                x='total_revenue', 
                y='product_category_name', 
                data=problematic_cats,
                palette='Reds_r', 
                hue='product_category_name', 
                legend=False, 
                ax=ax
            )
            
            ax.set_title('Top 5 Kategori Produk: Revenue Tertinggi namun Rata-rata Review < 3.0', fontsize=16, pad=20, fontweight='bold')
            ax.set_xticks([]) 
            ax.set_xlabel('') 
            ax.set_ylabel('Kategori Produk', fontsize=12, fontweight='bold', labelpad=10)
            sns.despine(bottom=True, right=True, top=True)
            
            for i, p in enumerate(ax.patches):
                revenue_val = p.get_width()
                avg_rev = problematic_cats['avg_review_score'].iloc[i]
                ax.annotate(
                    f' R$ {revenue_val:,.2f}  |  Avg ⭐: {avg_rev:.2f}', 
                    (revenue_val, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', xytext=(10, 0), textcoords='offset points',
                    fontsize=12, fontweight='medium', color='#333333'
                )
                
            ax.set_xlim(0, problematic_cats['total_revenue'].max() * 1.45)
            plt.tight_layout()
            st.pyplot(fig_q3)
        else:
            st.success("Bagus! Tidak ada produk berpendapatan tinggi dengan rating di bawah 3.0 pada rentang waktu ini.")
    else:
        st.info("Tidak ada data produk pada rentang waktu ini.")

st.caption('Copyright (c) Pyta Nur Chumairah 2026')