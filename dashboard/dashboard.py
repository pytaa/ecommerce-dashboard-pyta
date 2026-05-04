# import libraries
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, "all_data.csv")
df = pd.read_csv(file_path)
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
    
    # ============================================================================
    # Q1: SELLERS - Top 10 Penjual Barang Berat dengan Rating > 4.0
    # ============================================================================
    st.subheader("Top Performers: Heavy Goods Sellers (> 7kg)")
    heavy_df = main_df[main_df['product_weight_g'] > 7000].copy()
    
    if not heavy_df.empty:
        # Agregasi: seller dengan barang > 7kg, groupby seller_id
        seller_metrics = heavy_df.groupby('seller_id').agg({
            'order_id': 'nunique',
            'review_score': 'mean'
        }).reset_index()
        seller_metrics.columns = ['seller_id', 'total_heavy_orders', 'avg_review_score']
        
        # Filter: rating > 4.0
        high_performing_sellers = seller_metrics[seller_metrics['avg_review_score'] > 4.0] \
            .sort_values(by='total_heavy_orders', ascending=False)
        
        if not high_performing_sellers.empty:
            top_10_sellers = high_performing_sellers.head(10).copy()
            # Truncate seller_id ke 8 karakter + "..."
            top_10_sellers['seller_id_short'] = top_10_sellers['seller_id'].str[:8] + "..."

            fig_sellers, ax = plt.subplots(figsize=(10, 6))
            
            sns.barplot(
                x='total_heavy_orders',
                y='seller_id_short',
                data=top_10_sellers,
                palette='Blues_r',
                hue='seller_id_short',
                legend=False,
                ax=ax
            )
            
            ax.set_title('Top 10 Sellers Barang Berat (>7kg) dengan Skor > 4.0', 
                        fontsize=15, pad=20, fontweight='bold')
            ax.set_xlabel('Jumlah Pesanan yang Berhasil', fontsize=12)
            ax.set_ylabel('Seller ID (Original)', fontsize=12)
            
            # Hilangkan sumbu X
            ax.set_xticks([])
            sns.despine(bottom=True, left=False)
            
            # Anotasi gabungan: "Jumlah Pesanan | Avg ★"
            for i, p in enumerate(ax.patches):
                count = int(p.get_width())
                avg_score = top_10_sellers['avg_review_score'].iloc[i]
                
                ax.annotate(
                    f' {count} Pesanan  |  Avg ★: {avg_score:.2f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left',
                    va='center',
                    xytext=(8, 0),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    color='#333333'
                )
            
            ax.set_xlim(0, top_10_sellers['total_heavy_orders'].max() * 1.3)
            
            plt.tight_layout()
            st.pyplot(fig_sellers)
        else:
            st.info("Tidak ada seller yang memenuhi kriteria (Rating > 4.0) pada rentang waktu ini.")
    else:
        st.info("Tidak ada transaksi barang berat (>7kg) pada rentang waktu ini.")

    st.divider()
    
    # ============================================================================
    # Q2: LOGISTICS - Tren Keterlambatan di Sao Paulo + Kategori Terburuk
    # ============================================================================
    st.subheader("Sao Paulo Logistics Analysis (Q2)")
    sp_df = main_df[main_df['customer_city'] == 'sao paulo'].copy()
    
    if not sp_df.empty:
        # Prepare data: convert datetime dan hitung pure_delay_days
        sp_df['order_delivered_customer_date'] = pd.to_datetime(sp_df['order_delivered_customer_date'])
        sp_df['order_estimated_delivery_date'] = pd.to_datetime(sp_df['order_estimated_delivery_date'])
        sp_df['delivery_margin_days'] = (sp_df['order_delivered_customer_date'] - sp_df['order_estimated_delivery_date']).dt.days
        sp_df['pure_delay_days'] = sp_df['delivery_margin_days'].clip(lower=0)
        sp_df['purchase_month'] = sp_df['order_purchase_timestamp'].dt.month
        
        # Agregasi untuk tren bulanan
        monthly_trend = sp_df.groupby('purchase_month')[['pure_delay_days']].mean().reset_index()
        
        # Agregasi untuk kategori terburuk (min 30 pesanan)
        category_analysis = sp_df.groupby('product_category_name').agg({
            'pure_delay_days': 'mean',
            'order_id': 'nunique'
        }).reset_index()
        category_analysis.columns = ['product_category_name', 'avg_delay_days', 'total_order_cases']
        
        top_10_filtered = category_analysis[category_analysis['total_order_cases'] >= 30] \
            .sort_values(by='avg_delay_days', ascending=False) \
            .head(10)
        
        # Subplots: 2 baris, 1 kolom
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        
        # Plot 1: Line chart - Tren bulanan
        sns.lineplot(
            x='purchase_month',
            y='pure_delay_days',
            data=monthly_trend,
            marker='o',
            color='crimson',
            linewidth=2.5,
            markersize=8,
            ax=ax[0]
        )
        ax[0].set_title('Tren Rata-rata Keterlambatan Murni di Sao Paulo', 
                       fontsize=15, pad=15, fontweight='bold')
        ax[0].set_xlabel('Bulan Pembelian', fontsize=12)
        ax[0].set_ylabel('Rata-rata Terlambat (Hari)', fontsize=12)
        ax[0].set_xticks(range(1, 7))
        ax[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun'])
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Bar chart - Kategori terburuk (min 30 pesanan)
        if not top_10_filtered.empty:
            sns.barplot(
                x='avg_delay_days',
                y='product_category_name',
                data=top_10_filtered,
                palette='Reds_r',
                hue='product_category_name',
                legend=False,
                ax=ax[1]
            )
            ax[1].set_title('Top 10 Kategori Produk dengan Rata-rata Keterlambatan Terburuk (Min. 30 Pesanan)', 
                           fontsize=15, pad=15, fontweight='bold')
            ax[1].set_xlabel('Rata-rata Keterlambatan (Hari)', fontsize=12)
            ax[1].set_ylabel('Kategori Produk', fontsize=12)
            
            # Anotasi: tampilkan nilai hari di ujung bar
            for p in ax[1].patches:
                ax[1].annotate(
                    f' {p.get_width():.1f} Hari',
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left',
                    va='center',
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=11
                )
            
            ax[1].set_xlim(0, top_10_filtered['avg_delay_days'].max() * 1.2)
        else:
            ax[1].text(0.5, 0.5, 'Tidak ada kategori dengan minimum 30 pesanan', 
                      ha='center', va='center', transform=ax[1].transAxes)
            ax[1].set_xticks([])
            ax[1].set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Tidak ada pesanan dari Sao Paulo pada rentang waktu ini.")

    st.divider()
    
    # ============================================================================
    # Q3: PRODUCTS - Kategori Revenue Tinggi dengan Rating < 3.0
    # ============================================================================
    st.subheader("High Revenue with Low Satisfaction")
    
    # Agregasi: revenue dan average rating per kategori produk
    category_metrics = main_df.groupby('product_category_name').agg({
        'price': 'sum',
        'review_score': 'mean'
    }).reset_index()
    category_metrics.columns = ['product_category_name', 'total_revenue', 'avg_review_score']
    
    # Hitung volume (jumlah item terjual) per kategori
    volume_df = main_df.groupby('product_category_name').agg({
        'order_item_id': 'count'
    }).reset_index()
    volume_df.columns = ['product_category_name', 'total_volume']
    
    # Merge metrics dan volume
    category_performance = category_metrics.merge(volume_df, on='product_category_name', how='left')
    
    # Filter: rating < 3.0 DAN volume >= 30
    threshold_volume = 30
    threshold_rating = 3.0
    
    problematic_cats = category_performance[
        (category_performance['avg_review_score'] < threshold_rating) &
        (category_performance['total_volume'] >= threshold_volume)
    ].sort_values(by='total_revenue', ascending=False)
    
    if not problematic_cats.empty:
        top_5_problematic = problematic_cats.head(5).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            x='total_revenue',
            y='product_category_name',
            data=top_5_problematic,
            palette='Reds_r',
            hue='product_category_name',
            legend=False,
            ax=ax
        )
        
        ax.set_title('Top 5 Kategori: Revenue Tertinggi namun Rata-rata Review < 3.0',
                    fontsize=15, pad=20, fontweight='bold')
        
        # Hilangkan sumbu X
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel('Kategori Produk', fontsize=12, fontweight='bold', labelpad=10)
        
        sns.despine(bottom=True, right=True, top=True)
        
        # Anotasi gabungan: "Revenue | Avg ★"
        for i, p in enumerate(ax.patches):
            revenue_val = p.get_width()
            avg_rev = top_5_problematic['avg_review_score'].iloc[i]
            
            ax.annotate(
                f' R$ {revenue_val:,.0f}  |  Avg ★: {avg_rev:.2f}',
                (revenue_val, p.get_y() + p.get_height() / 2.),
                ha='left',
                va='center',
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color='#333333'
            )
        
        ax.set_xlim(0, top_5_problematic['total_revenue'].max() * 1.45)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.success("Bagus! Tidak ada kategori dengan rating < 3.0 pada rentang waktu ini.")

st.caption('Copyright (c) Pyta Nur Chumairah 2026')