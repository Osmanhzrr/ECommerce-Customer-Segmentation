"""
=============================================================================
PROJE: E-Ticare Omni-Channel Satış Analizi, RFM & K-Means Müşteri Kümeleme
DOSYA: Churn.py
AÇIKLAMA:
    -Faz 1: Veri birleştirme ve bellek optimizasyonu.
    -Faz 2: Ham veri üzerinden iade (Return), Misafir(Guest) ve Genel Ciro analizi.
    -Faz 3: Makine Öğrenmesi (RFM) için verinin temizlenmesi (Filtreleme).
    -Faz 4: Kural Bazlı RFM (Recency, Frequency, Monetary) Skorlaması.
    -Faz 5: Logaritmik K-Means (K=4) Kümeleme.
    -Faz 6: Model Doğrulaması ve Nihai Müşteri Segmentasyon Grafikleri (4 BI Grafiği).
=============================================================================
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Görsel ve Terminal Ayarları
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ==============================================================================
# FAZ 1: VERİ İÇERİ AKTARMA & BELLEK OPTİMİZASYONU
# ==============================================================================
print("1. Veriler yükleniyor ve bellek optimizasyonu yapılıyor...")
df_2009 = pd.read_csv("Year 2009-2010.csv", encoding='iso-8859-9')
df_2010 = pd.read_csv("Year 2010-2011.csv", encoding='iso-8859-9')
df_raw = pd.concat([df_2009, df_2010], ignore_index=True)

#NOT: Verinin boş sütun analizleri, eksik veriler ve diğer columns gibi özellikleri bilinmektedir ekstradan df.columns gibi kodlar yazılmamıştır


# Bellek kullanımını azaltmak için tip dönüşümleri
df_raw = df_raw.astype({
    'Invoice': 'category',
    'StockCode': 'category',
    'Description': 'string',
    'Country': 'category'
})

df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
df_raw['Customer ID'] = df_raw['Customer ID'].astype('Int64')
df_raw['Total_Price'] = df_raw['Quantity'] * df_raw['Price']

# ==============================================================================
# FAZ 2: GENEL SATIŞ ANALİZİ VE İADE METRİKLERİ (MAKRO EDA)
# ==============================================================================
print("\n" + "="*70)
print("FAZ 2: ŞİRKET GENEL MAKRO SATIŞ VE İADE RAPORU")
print("="*70)

# İade ve Normal İşlem Ayrımı
iadeler = df_raw[df_raw['Invoice'].astype(str).str.contains('C', na=False)]
basarili_islemler = df_raw[~df_raw['Invoice'].astype(str).str.contains('C', na=False)]

# Kayıtlı ve Misafir Ayrımı (Sadece Başarılı İşlemler için)
kayitli_satislar = basarili_islemler[basarili_islemler['Customer ID'].notna()]
misafir_satislar = basarili_islemler[basarili_islemler['Customer ID'].isna()]

# Makro Finansal Metrikler
brut_ciro = basarili_islemler[basarili_islemler['Total_Price'] > 0]['Total_Price'].sum()
iade_zarari = abs(iadeler['Total_Price'].sum())# Eksi değeri mutlak değere çeviriyoruz
kayitli_ciro = kayitli_satislar[kayitli_satislar['Total_Price'] > 0]['Total_Price'].sum()
misafir_ciro = misafir_satislar[misafir_satislar['Total_Price'] > 0]['Total_Price'].sum()

print(f"Toplam İşlem Hacmi       : {len(df_raw):,} Satır")
print(f"Toplam Brüt Ciro         : £{brut_ciro:,.2f}")
print(f"Toplam İade Tutarı       : £{iade_zarari:,.2f} (Cironun %{(iade_zarari/brut_ciro)*100:.1f}'i)")
print("-" * 70)
print(f"Kayıtlı Müşteri Cirosu   : £{kayitli_ciro:,.2f} (%{kayitli_ciro/brut_ciro*100:.1f})")
print(f"Misafir (Guest) Cirosu   : £{misafir_ciro:,.2f} (%{misafir_ciro/brut_ciro*100:.1f})")
print("="*70)

# --- MAKRO GRAFİKLER (ŞİRKET GENEL BAKIŞ) ---
print("Birinci Makro Dashboard (macro_sales_dashboard1.png) oluşturuluyor...")
fig, axes = plt.subplots(2, 2, figsize=(18,22))
fig.suptitle('Şirket Makro Performans Analizi (Satış, İade ve Müşteri Kanalları)', fontsize=16, fontweight='bold')

# Grafik 1: Aylık Ciro Trendi
basarili_islemler_trend = basarili_islemler.copy()
basarili_islemler_trend['YearMonth'] = basarili_islemler_trend['InvoiceDate'].dt.to_period('M')
aylik_ciro = basarili_islemler_trend.groupby('YearMonth')['Total_Price'].sum()
aylik_ciro.index = aylik_ciro.index.astype(str)
sns.lineplot(x=aylik_ciro.index, y=aylik_ciro.values, marker='o', ax=axes[0,0], color='#2980b9', linewidth=2)
axes[0, 0].set_title('Aylık Brüt Ciro Trendi', fontweight='bold')
axes[0, 0].tick_params(axis='x',rotation=45)
axes[0, 0].set_ylabel('Ciro (£)')

# Grafik 2: Kayıtlı vs Misafir Ciro (Donut)
axes[0, 1].pie(
    [kayitli_ciro, misafir_ciro],
    labels=['Kayıtlı Üyeler', 'Misafir (Guest)'],
    autopct='%1.1f%%',
    colors=['#27ae60', '#f39c12'],
    startangle=90
)
axes[0, 1].add_artist(plt.Circle((0,0), 0.65, fc='white'))
axes[0, 1].set_title('Kayıtlı ve Misafir Kullanıcı Ciro Dağılımı', fontweight='bold')

# Grafik 3: Brüt Ciro vs İade Edilen Tutar
sns.barplot(x=['Brüt Ciro', 'İade (Kayıp)'], y=[brut_ciro, iade_zarari], ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])
axes[1, 0].set_title('Brüt Gelir ve İade Zararı Karşılaştırması', fontweight='bold')
axes[1, 0].set_ylabel('Tutar (£)')

# Grafik 4: En Çok İade Edilen İlk 5 Ürün
en_cok_iade = iadeler.groupby('Description')['Quantity'].sum().abs().sort_values(ascending=False).head(5)
sns.barplot(y=en_cok_iade.index, x=en_cok_iade.values, ax=axes[1, 1], palette='Reds_r')
axes[1, 1].set_title('En Çok İade Edilen İlk 5 Ürün (Adet)', fontweight='bold')
axes[1, 1].set_xlabel('İade Adedi')

plt.tight_layout()
plt.savefig('macro_sales_dashboard1.png', dpi=300)
plt.close()

# ==============================================================================
# FAZ 2.1: İLERİ DÜZEY MAKRO SATIŞ ANALİZİ (DASHBOARD 2)
# ==============================================================================
print("İkinci Makro Dashboard (macro_sales_dashboard2.png) oluşturuluyor...")
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 12))
fig3.suptitle('Şirket Makro Performans Analizi 2 (Zaman, Lokasyon ve Sepet Dinamikleri)', fontsize=16, fontweight='bold')

# Tarih türetimlerini ana tabloya güvenli bir şekilde ekleyelim
basarili_islemler = basarili_islemler.copy()
basarili_islemler['YearMonth'] = basarili_islemler['InvoiceDate'].dt.to_period('M')
basarili_islemler['DayOfWeek'] = basarili_islemler['InvoiceDate'].dt.day_name()
basarili_islemler['Hour'] = basarili_islemler['InvoiceDate'].dt.hour

# Grafik 1: Gün ve Saat Bazlı Sipariş Yoğunluğu (Heatmap)
gunler = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = basarili_islemler.groupby(['DayOfWeek', 'Hour'])['Invoice'].nunique().unstack().fillna(0)
heatmap_data = heatmap_data.reindex([g for g in gunler if g in heatmap_data.index])

sns.heatmap(heatmap_data, cmap='YlGnBu', ax=axes3[0, 0], annot=False, linewidths=.5)
axes3[0, 0].set_title('Gün ve Saat Bazlı Sipariş Yoğunluğu', fontweight='bold')
axes3[0, 0].set_xlabel('Günün Saatleri')
axes3[0, 0].set_ylabel('Haftanın Günleri')

# Grafik 2: En Çok Ciro Getiren İlk 10 Ülke (Bar Chart)
# observed=True ile kullanılmayan kategorik değerlerin getireceği hatalar önlenir
ulke_ciro = (
    basarili_islemler.groupby('Country', observed=True)['Total_Price']
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
sns.barplot(x=ulke_ciro.values, y=ulke_ciro.index.astype(str), ax=axes3[0, 1], palette='magma')
axes3[0, 1].set_title('En Çok Ciro Getiren İlk 10 Ülke', fontweight='bold')
axes3[0, 1].set_xlabel('Toplam Ciro (£)')
axes3[0, 1].set_ylabel('')

# Grafik 3: En Çok Ciro Getiren 10 Ürün (Lokomotif Ürünler)
urun_ciro = (
    basarili_islemler.groupby('Description')['Total_Price']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
sns.barplot(x=urun_ciro.values, y=urun_ciro.index.astype(str), ax=axes3[1, 0], palette='crest')
axes3[1, 0].set_title('Kasaya En Çok Nakit Bırakan 10 Ürün (Lokomotifler)', fontweight='bold')
axes3[1, 0].set_xlabel('Toplam Ciro (£)')
axes3[1, 0].set_ylabel('')

# Grafik 4: Aylık Ortalama Sepet Tutarı (AOV) Trendi
aylik_sepet = basarili_islemler.groupby('YearMonth').agg(
    Toplam_Ciro=('Total_Price', 'sum'),
    Fatura_Sayisi=('Invoice', 'nunique')
)
aylik_sepet['AOV'] = aylik_sepet['Toplam_Ciro'] / aylik_sepet['Fatura_Sayisi']
aylik_sepet.index = aylik_sepet.index.astype(str)

sns.lineplot(x=aylik_sepet.index, y=aylik_sepet['AOV'], ax=axes3[1, 1], marker='s', color='#8e44ad', linewidth=2.5)
axes3[1, 1].set_title('Aylık Ortalama Sepet Tutarı (AOV) Trendi', fontweight='bold')
axes3[1, 1].tick_params(axis='x', rotation=45)
axes3[1, 1].set_ylabel('Ortalama Sepet Tutarı (£)')
axes3[1, 1].set_xlabel('Aylar')

plt.tight_layout()
plt.savefig('macro_sales_dashboard2.png', dpi=300)
plt.close()

# ==============================================================================
# FAZ 3: MAKİNE ÖĞRENMESİ (ML) İÇİN VERİ TEMİZLİĞİ
# ==============================================================================
# RFM sadece kayıtlı müşteriler ve başarılı işlemlerle yapılır
df_ml = df_raw[~df_raw['Invoice'].astype(str).str.contains('C', na=False)].copy()
df_ml = df_ml[(df_ml['Quantity'] > 0) & (df_ml['Price'] > 0)]
df_ml = df_ml.dropna(subset=["Description", "Customer ID"])

# ==============================================================================
# FAZ 4: KURAL BAZLI RFM METRİKLERİ VE SEGMENTASYON
# ==============================================================================
analiz_tarihi = df_ml['InvoiceDate'].max() + dt.timedelta(days=2)

rfm = df_ml.groupby('Customer ID').agg({
    'InvoiceDate': lambda date: (analiz_tarihi - date.max()).days,
    'Invoice': 'nunique',
    'Total_Price': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

rfm['recency_score'] = pd.qcut(rfm['Recency'].rank(method="first"), 5, labels=[5,4,3,2,1])
rfm['frequency_score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['Monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm['RFM_SCORE'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}
rfm['Segment_RuleBased'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# ==============================================================================
# FAZ 5: LOGARİTMİK K-MEANS KÜMELEME (UNSUPERVISED ML)
# ==============================================================================
rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# K-Means'da n_clusters=4 seçildi çünkü elbow yönteminde 4 en mantıklı kümeleme sayısı.
# Hiperparametre Optimizasyonu ile K-Means
# max_iter: Algoritmanın merkezleri bulmak için yapacağı maksimum deneme (Varsayılan 300'dür, 500 yaptık)
# tol: Merkezlerin kayma toleransı (Daha hassas bir durma noktası belirledik)
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
    max_iter=500,
    tol=1e-4
)
rfm['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled) + 1

cluster_names = {
    1: 'Küme 1: Pasifler / Uykuda',
    2: 'Küme 2: Şampiyonlar & VIP',
    3: 'Küme 3: Risktekiler (Eski Sadıklar)',
    4: 'Küme 4: Potansiyel Yeniler'
}
rfm['Cluster_Name'] = rfm['KMeans_Cluster'].map(cluster_names)

# ==============================================================================
# FAZ 6: BI GÖRSELLEŞTİRME VE ÇIKTILAR (K-MEANS & RFM)
# ==============================================================================
print("\n" + "="*70)
print("FAZ 6: KURAL BAZLI (RFM) VE MAKİNE ÖĞRENMESİ (K-MEANS) ÇAPRAZ TABLOSU")
print("="*70)
print(pd.crosstab(rfm['Segment_RuleBased'], rfm['Cluster_Name']))
# --- ML GRAFİKLERİ (RFM & K-MEANS GÖRSELLEŞTİRMESİ) ---
palette_ml = {1: '#e74c3c', 2: '#2ecc71', 3: '#f39c12', 4: '#3498db'}

fig2 = plt.figure(figsize=(18, 12))
fig2.suptitle('K-Means Müşteri Davranış Analizi ve RFM Doğrulaması', fontsize=16, fontweight='bold')

# Grafik 1: 3D Logaritmik K-Means Dağılımı
ax1 = fig2.add_subplot(221, projection='3d')
for c_id in sorted(rfm['KMeans_Cluster'].unique()):
    sub = rfm[rfm['KMeans_Cluster'] == c_id]
    ax1.scatter(np.log1p(sub['Recency']), np.log1p(sub['Frequency']), np.log1p(sub['Monetary']),
                c=palette_ml[c_id], label=cluster_names[c_id], s=25, alpha=0.7)
ax1.set_title('3D Log-Scaled Müşteri Kümeleri', fontweight='bold')
ax1.set_xlabel('Log(Recency)')
ax1.set_ylabel('Log(Frequency)')
ax1.set_zlabel('Log(Monetary)')
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=8)

# Grafik 2: Kümelerin Müşteri Hacmi (Donut)
ax2 = fig2.add_subplot(222)
cluster_counts = rfm['KMeans_Cluster'].value_counts().sort_index()
ax2.pie(cluster_counts, labels=[cluster_names[i] for i in cluster_counts.index], 
        autopct='%1.1f%%', colors=[palette_ml[i] for i in cluster_counts.index], startangle=140)
ax2.add_artist(plt.Circle((0,0), 0.65, fc='white'))
ax2.set_title('K-Means Küme Nüfus Dağılımı', fontweight='bold')

# Grafik 3: 2D Scatter (Recency vs Frequency - Logaritmik Eksende)
ax3 = fig2.add_subplot(223)
sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Cluster_Name', 
                palette=[palette_ml[i] for i in sorted(palette_ml.keys())], alpha=0.6, s=30, ax=ax3)
ax3.set_yscale('log')
ax3.set_title('Recency vs Frequency Ayrışması (Log Ölçeği)', fontweight='bold')

# Grafik 4: Kural Bazlı RFM (Segment_RuleBased) Bar Plot
ax4 = fig2.add_subplot(224)
segment_order = rfm['Segment_RuleBased'].value_counts().index
sns.countplot(data=rfm, y='Segment_RuleBased', order=segment_order, palette='viridis', ax=ax4)
ax4.set_title('Kural Bazlı Dağılım (Machine Learning ile Kıyas İçin)', fontweight='bold')
ax4.set_xlabel('Müşteri Sayısı')

plt.tight_layout()
plt.savefig('kmeans_rfm_dashboard.png', dpi=300)