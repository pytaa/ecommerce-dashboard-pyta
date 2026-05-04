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

    # =====================================================================
    # Section 1 (Q1): Top High-Performing Sellers untuk Barang Berat (>7kg)
    # Aggregation: filter product_weight_g > 7000, avg_review_score > 4.0,
    # group by seller_id, sort by total orders, top 10
    # Palette: Blues_r, annotations: " N Pesanan  |  Avg ★: X.XX"
    # =====================================================================
    st.subheader("Top Performers: Heavy Goods Sellers (> 7kg)")

    heavy_df = main_df[main_df['product_weight_g'] > 7000].copy()

    if not heavy_df.empty:
        # Agregasi performa seller (sesuai notebook)
        seller_performance = heavy_df.groupby('seller_id').agg(
            avg_review_score=('review_score', 'mean'),
            total_heavy_orders=('order_id', 'nunique')
        ).reset_index()

        # Filter skor > 4.0 dan urutkan dari pesanan terbanyak
        high_performing_sellers = seller_performance[
            seller_performance['avg_review_score'] > 4.0
        ].sort_values(by='total_heavy_orders', ascending=False)

        top_10_sellers = high_performing_sellers.head(10).copy()

        if not top_10_sellers.empty:
            # Truncate seller_id: 8 karakter pertama + "..."
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

            # Menghilangkan sumbu X
            ax.set_xticks([])
            sns.despine(bottom=True, left=False, ax=ax)

            # Annotations: " N Pesanan  |  Avg ★: X.XX"
            for i, p in enumerate(ax.patches):
                count = int(p.get_width())
                avg_score = top_10_sellers['avg_review_score'].iloc[i]
                ax.annotate(
                    f' {count} Pesanan  |  Avg \u2605: {avg_score:.2f}',
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
            st.info("Tidak ada seller yang memenuhi kriteria (Skor > 4.0) pada rentang waktu ini.")
    else:
        st.info("Tidak ada transaksi barang berat (>7kg) pada rentang waktu ini.")

    st.divider()

    # =====================================================================
    # Section 2 (Q2): Sao Paulo Logistics Analysis (Jan-Jun 2018)
    # Aggregation:
    #   - Filter: customer_city == 'sao paulo', Jan-Jun 2018
    #   - pure_delay_days = clip(delivery_margin_days, lower=0)
    #   - Monthly trend: group by purchase_month, mean of pure_delay_days
    #   - Top 10 worst categories: group by product_category_name,
    #       mean of pure_delay_days, filter total_order_cases >= 30, top 10
    # Palette: Reds_r for bar chart, annotations: " X.X Hari"
    # =====================================================================
    st.subheader("Sao Paulo Logistics Analysis (January-June 2018)")

    sp_df = main_df[
        (main_df['customer_city'] == 'sao paulo') &
        (main_df['order_purchase_timestamp'] >= '2018-01-01') &
        (main_df['order_purchase_timestamp'] <= '2018-06-30')
    ].copy()

    if not sp_df.empty:
        sp_df['order_delivered_customer_date'] = pd.to_datetime(sp_df['order_delivered_customer_date'])
        sp_df['order_estimated_delivery_date'] = pd.to_datetime(sp_df['order_estimated_delivery_date'])

        sp_df['delivery_margin_days'] = (
            sp_df['order_delivered_customer_date'] - sp_df['order_estimated_delivery_date']
        ).dt.days

        sp_df['pure_delay_days'] = sp_df['delivery_margin_days'].clip(lower=0)
        sp_df['purchase_month'] = sp_df['order_purchase_timestamp'].dt.month

        # Monthly trend
        monthly_trend = sp_df.groupby('purchase_month')[['pure_delay_days', 'delivery_margin_days']].mean().reset_index()

        # Category analysis: worst categories with min 30 orders
        category_analysis = sp_df.groupby('product_category_name').agg(
            avg_delay_days=('pure_delay_days', 'mean'),
            total_order_cases=('order_id', 'nunique')
        ).reset_index()

        top_10_filtered = category_analysis[category_analysis['total_order_cases'] >= 30] \
            .sort_values(by='avg_delay_days', ascending=False) \
            .head(10)

        # --- Plot: 2 subplots ---
        fig_logistics, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        # Visualisasi 1: Tren Bulanan
        month_labels = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mei', 6: 'Jun'}
        sns.lineplot(
            x='purchase_month',
            y='pure_delay_days',
            data=monthly_trend,
            marker='o',
            color='crimson',
            linewidth=2.5,
            markersize=8,
            ax=axs[0]
        )
        axs[0].set_title('Tren Rata-rata Keterlambatan Murni di Sao Paulo (Jan-Jun 2018)',
                          fontsize=13, pad=15, fontweight='bold')
        axs[0].set_xlabel('Bulan Pembelian', fontsize=12)
        axs[0].set_ylabel('Rata-rata Terlambat (Hari)', fontsize=12)
        axs[0].set_xticks(monthly_trend['purchase_month'].tolist())
        axs[0].set_xticklabels([month_labels.get(m, str(m)) for m in monthly_trend['purchase_month'].tolist()])
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Visualisasi 2: Kategori Terburuk
        if not top_10_filtered.empty:
            sns.barplot(
                x='avg_delay_days',
                y='product_category_name',
                data=top_10_filtered,
                palette='Reds_r',
                hue='product_category_name',
                legend=False,
                ax=axs[1]
            )
            axs[1].set_title('Top 10 Kategori Produk dengan Rata-rata Keterlambatan Terburuk (Min. 30 Pesanan)',
                              fontsize=13, pad=15, fontweight='bold')
            axs[1].set_xlabel('Rata-rata Keterlambatan (Hari)', fontsize=12)
            axs[1].set_ylabel('Kategori Produk', fontsize=12)

            # Label Angka di samping bar
            for p in axs[1].patches:
                axs[1].annotate(
                    f' {p.get_width():.1f} Hari',
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=11
                )
            axs[1].set_xlim(0, top_10_filtered['avg_delay_days'].max() * 1.2)
        else:
            axs[1].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_logistics)
    else:
        st.info("Tidak ada pesanan dari Sao Paulo pada periode Januari-Juni 2018 dalam rentang waktu yang dipilih.")

    st.divider()

    # =====================================================================
    # Section 3 (Q3): High Revenue with Low Satisfaction (Jan-Jun 2018)
    # Aggregation:
    #   - Filter: Jan-Jun 2018
    #   - group by product_category_name: total_revenue (sum price),
    #       avg_review_score (mean), total_volume (count order_item_id)
    #   - Filter: avg_review_score < 4.0 AND total_volume >= 30
    #   - Sort by total_revenue DESC, top 5
    # Palette: Reds_r, annotations: " $X,XXX.XX  |  Avg ★: X.XX"
    # X-axis hidden, sns.despine(bottom=True, right=True, top=True)
    # =====================================================================
    st.subheader("High Revenue with Low Satisfaction (Jan-Jun 2018)")

    period_df = main_df[
        (main_df['order_purchase_timestamp'] >= '2018-01-01') &
        (main_df['order_purchase_timestamp'] < '2018-07-01')
    ].copy()

    if not period_df.empty:
        # Category metrics: revenue & review score
        category_metrics = period_df.groupby('product_category_name').agg(
            total_revenue=('price', 'sum'),
            avg_review_score=('review_score', 'mean')
        ).reset_index()

        # Volume per category
        volume_df = period_df.groupby('product_category_name').agg(
            total_volume=('order_item_id', 'count')
        ).reset_index()

        category_performance = category_metrics.merge(volume_df, on='product_category_name', how='left')

        # Filter: avg_review_score < 4.0 AND total_volume >= 30
        problematic_cats = category_performance[
            (category_performance['avg_review_score'] < 4.0) &
            (category_performance['total_volume'] >= 30)
        ].sort_values(by='total_revenue', ascending=False)

        top_5_problematic = problematic_cats.head(5).reset_index(drop=True)

        if not top_5_problematic.empty:
            fig_products, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(
                x='total_revenue',
                y='product_category_name',
                data=top_5_problematic,
                palette='Reds_r',
                hue='product_category_name',
                legend=False,
                ax=ax
            )

            ax.set_title(
                'Top 5 Kategori Produk: Revenue Tertinggi namun Rata-rata Review < 3.0\n(Periode Januari - Juni 2018)',
                fontsize=14, pad=20, fontweight='bold'
            )

            # Menyembunyikan sumbu X
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('Kategori Produk', fontsize=12, fontweight='bold', labelpad=10)

            sns.despine(bottom=True, right=True, top=True, ax=ax)

            # Annotations: " $Revenue  |  Avg ★: X.XX"
            for i, p in enumerate(ax.patches):
                revenue_val = p.get_width()
                avg_rev = top_5_problematic['avg_review_score'].iloc[i]
                ax.annotate(
                    f' ${revenue_val:,.2f}  |  Avg \u2605: {avg_rev:.2f}',
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
            st.pyplot(fig_products)
        else:
            st.success("Bagus! Tidak ada produk dengan revenue tinggi dan rating di bawah 4.0 pada periode Januari-Juni 2018.")
    else:
        st.info("Tidak ada data pada periode Januari-Juni 2018 dalam rentang waktu yang dipilih.")

st.caption('Copyright (c) Pyta Nur Chumairah 2026')