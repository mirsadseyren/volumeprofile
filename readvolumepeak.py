import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_volume_profile(ticker="INVES", period="2d", interval="1m", bins=50):
    print(f"{ticker} için {period} periyotlu, {interval} aralıklı veri çekiliyor...")
    
    # Veri Çekme
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if data.empty:
        print(f"Hata: {ticker} için veri bulunamadı veya bu periyot/aralık geçerli değil.")
        print("Not: Borsa İstanbul hisseleri için sonuna '.IS' eklemeyi unutmayın (Örn: 'INVEO.IS' veya 'INVES.IS').")
        return
        
    print(f"Toplam {len(data)} satır veri çekildi.")
    
    # yfinance son sürümlerinde sütunlar MultiIndex (Ticker ile) dönebiliyor
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.levels[1]:
            close_prices = data['Close'][ticker].dropna()
            volumes = data['Volume'][ticker].dropna()
        else:
            close_prices = data['Close'].iloc[:, 0].dropna()
            volumes = data['Volume'].iloc[:, 0].dropna()
    else:
        close_prices = data['Close'].dropna()
        volumes = data['Volume'].dropna()
        
    # Fiyat aralığını belirle
    min_price = close_prices.min()
    max_price = close_prices.max()
    
    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        print("Fiyat verisi anomali içeriyor veya hiç dalgalanma yok, çizilemiyor.")
        return

    # Fiyatları belirli sayıda gruba (bin) ayır
    price_bins = np.linspace(min_price, max_price, bins)
    
    # Hangi fiyatın hangi gruba düştüğünü bul
    # indices, her değerin hangi bin aralığına denk geldiğini döner
    indices = np.digitize(close_prices, price_bins)
    
    # Her bir grup için toplam hacmi hesapla
    volume_profile = np.zeros(bins)
    for i in range(len(close_prices)):
        idx = indices[i] - 1  # np.digitize 1-tabanlı indeks döner, bunu 0-tabanlı yapıyoruz
        if idx >= bins: 
            idx = bins - 1
        elif idx < 0:
            idx = 0
            
        # Pandas Series'ten hacim verisini iloc kullanarak al
        volume_profile[idx] += float(volumes.iloc[i])
        
    # En yüksek hacimli fiyatı (Point of Control - POC) bul
    poc_index = np.argmax(volume_profile)
    poc_price = price_bins[poc_index]
    
    # Grafiği Çiz
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Pazarın kapalı olduğu saatlerdeki boşluğu kaldırmak için X eksenini sıralı integer yapıyoruz
    x_indices = np.arange(len(close_prices))
    
    # 1) Sol tarafta Fiyat Grafiği (Zaman ekseninde Çizgi)
    ax1.plot(x_indices, close_prices, color='royalblue', linewidth=1.5, label='Kapanış Fiyatı')
    ax1.set_xlabel('Zaman (Gün İçi)')
    ax1.set_ylabel('Fiyat', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    
    # X ekseninde tarih/saat gösterimini düzelt
    num_ticks = 10
    step = max(1, len(x_indices) // num_ticks)
    tick_indices = x_indices[::step]
    
    if len(close_prices) > 0 and close_prices.index[0].date() != close_prices.index[-1].date():
        tick_labels = [close_prices.index[i].strftime("%d/%m %H:%M") for i in tick_indices]
    else:
        tick_labels = [close_prices.index[i].strftime("%H:%M") for i in tick_indices]
        
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45)
    
    # 2) Sağ tarafta Hacim Profili (Yatay Barlar - Şeffaf)
    ax2 = ax1.twiny()
    
    # Yüksekliği belirle
    bar_height = (max_price - min_price) / bins * 0.8
    
    # Renkleri belirle (POC'u kırmızı, diğerlerini gri/mavi yapalım)
    colors = ['lightsteelblue'] * bins
    colors[poc_index] = 'darkred'
    
    ax2.barh(price_bins, volume_profile, height=bar_height, 
             color=colors, alpha=0.4, edgecolor='none')
             
    # POC Çizgisi
    ax1.axhline(poc_price, color='darkred', linestyle='--', alpha=0.8, 
                label=f'POC (En Yoğun Fiyat): {poc_price:.2f}')
    
    ax2.set_xlabel('Hacim Toplamı (Volume Profile)', color='slategray')
    ax2.tick_params(axis='x', labelcolor='slategray')
    
    # Görünüm Ayarları
    # Eğer birden fazla gün varsa başlıkta her iki tarihi de (ilk ve son) gösterelim
    date_str = f"{data.index[0].date()} - {data.index[-1].date()}" if len(data) > 0 and data.index[0].date() != data.index[-1].date() else f"{data.index[-1].date()}"
    ax1.set_title(f'{ticker} Fiyat Hareketi ve Volume Profile (1m) | {date_str}', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Legend birleştirme
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Eğer "INVES" Türkiye piyasasından Inveo ise "INVEO.IS" olmalıdır.
    # Kodun hatasız çalışması için "INVES.IS" (Investco Holding) olarak ayarlıyoruz.
    plot_volume_profile(ticker="fonet.IS", period="5d", interval="1m", bins=50)
