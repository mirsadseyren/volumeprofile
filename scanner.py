import yfinance as yf
import pandas as pd
import numpy as np
import os
import time

def read_tickers(filename="stox.txt"):
    tickers = []
    if not os.path.exists(filename):
        print(f"{filename} bulunamadı!")
        return tickers
        
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Başlıkları, boş satırları veya normal ticker formatına uymayan uzun cümleleri atla
            if not line or len(line) > 7 or " " in line or line.startswith("Yüklediğin"):
                continue
            if line.startswith("İsteğin") or "Hisse" in line or line.startswith("Temizlenmiş") or line.startswith("İşte"):
                continue
                
            line = line.rstrip('.') # BINHO. gibi hatalı son harfleri temizle
            tickers.append(line + ".IS")
    return list(set(tickers))

def fetch_and_cache_data(tickers, cache_file="bist_data_cache.pkl"):
    print(f"Toplam {len(tickers)} hisse için 2 günlük 1 dakikalık veri çekiliyor...")
    
    # Çok fazla hisseyi tek seferde isiteyince yfinance timeout verebilir, 150'lik gruplar yapalım
    chunk_size = 150
    dfs = []
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"İndiriliyor: Grup {i//chunk_size + 1} / {(len(tickers) // chunk_size) + 1}...")
        try:
            data = yf.download(chunk, period="1d", interval="1m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    # Sadece Close ve Volume sütunlarını al
                    cols_to_keep = [col for col in data.columns if col[0] in ['Close', 'Volume']]
                    dfs.append(data[cols_to_keep])
                else: # Tek bir hisse ise (chunk_size=1 durumunda veya kalan son 1 hissede)
                    pass # Nadir bir durum, atlıyoruz basitlik için
        except Exception as e:
            print(f"Hata Group {i//chunk_size + 1}: {e}")
        time.sleep(0.5)
        
    print("Veri birleştiriliyor...")
    if dfs:
        final_df = pd.concat(dfs, axis=1)
        final_df.to_pickle(cache_file)
        print(f"Tüm veriler {cache_file} doysasına kaydedildi!")
        return final_df
    else:
        print("İndirme başarısız veya veri gelmedi.")
        return None

def calculate_poc(close_prices, volumes, bins=50):
    # Eksik verileri at (örneğin bazı hisseler o saatte işlem görmemiş olabilir)
    valid_data = pd.concat([close_prices, volumes], axis=1).dropna()
    if valid_data.empty:
        return None, None
        
    valid_close = valid_data.iloc[:, 0]
    valid_volume = valid_data.iloc[:, 1]
    
    min_price = valid_close.min()
    max_price = valid_close.max()
    
    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        return None, None
        
    price_bins = np.linspace(min_price, max_price, bins)
    indices = np.digitize(valid_close, price_bins)
    
    volume_profile = np.zeros(bins)
    for i in range(len(valid_close)):
        idx = indices[i] - 1
        if idx >= bins: idx = bins - 1
        elif idx < 0: idx = 0
        volume_profile[idx] += float(valid_volume.iloc[i])
        
    poc_index = np.argmax(volume_profile)
    poc_price = price_bins[poc_index]
    return poc_price, valid_close.iloc[-1]

def find_highest_poc_diff(cache_file="bist_data_cache.pkl"):
    if not os.path.exists(cache_file):
        print("Cache dosyası bulunamadı. Lütfen önce veri çekim işlemini tamamlayın.")
        return
        
    print(f"{cache_file} okunuyor ve analiz ediliyor...")
    data = pd.read_pickle(cache_file)
    
    results = []
    
    # MultiIndex yapısından hisse sembollerini çıkar
    # Örn sütunlar: ('Close', 'AKBNK.IS'), ('Volume', 'AKBNK.IS')
    try:
        tickers = sorted(list(set(data.columns.get_level_values(1))))
    except:
        print("Beklenmeyen veri yapısı, sütunları kontrol edin.")
        return

    for ticker in tickers:
        try:
            if ('Close', ticker) in data.columns and ('Volume', ticker) in data.columns:
                close_prices = data['Close'][ticker]
                volumes = data['Volume'][ticker]
                
                poc_price, last_price = calculate_poc(close_prices, volumes)
                
                if poc_price is not None and last_price is not None and last_price > 0:
                    # POC, fiyattan yukarıda olanı bulacağız (POC > Fiyat)
                    # Aradaki fark yüzdesini hesaplıyoruz: (POC - Fiyat) / Fiyat * 100
                    diff_pct = ((poc_price - last_price) / last_price) * 100
                    
                    if diff_pct > 0:
                        results.append({
                            'Ticker': ticker,
                            'LastPrice': last_price,
                            'POC': poc_price,
                            'DiffPct': diff_pct
                        })
        except Exception as e:
            continue
            
    if not results:
        print("Uygun hisse bulunamadı (Tüm hisselerin fiyatı POC'nin üzerinde veya hata oluştu).")
        return
        
    # Aradaki farkı en yüksek olanı sırala (POC'nin en tepede, fiyatın en dipte olduğu durumlar)
    results.sort(key=lambda x: x['DiffPct'], reverse=True)
    
    print("\n" + "="*60)
    print("📈 POC'Sİ FİYATTAN EN YUKARIDA OLAN HİSRELER (TOP 20)")
    print("="*60)
    print(f"{'Hisse':<12} | {'Son Fiyat':<10} | {'POC':<10} | {'Uzaklık Farkı(%)':<15}")
    print("-" * 60)
    for res in results[:20]:
        print(f"{res['Ticker']:<12} | {res['LastPrice']:<10.2f} | {res['POC']:<10.2f} | %{res['DiffPct']:.2f}")

if __name__ == "__main__":
    cache_file = "bist_data_cache.pkl"
    tickers_file = "stox.txt"
    
    tickers = read_tickers(tickers_file)
    print(f"'{tickers_file}' içerisinden {len(tickers)} geçerli hisse çıkarıldı.")
    
    if not os.path.exists(cache_file):
        fetch_and_cache_data(tickers, cache_file)
    else:
        print(len(tickers), "hisse listelendi.")
        print("💡 Önbellek dosyası bulundu. Yeniden indirilmeyecek. Eğer verileri yenilemek isterseniz 'bist_data_cache.pkl' dosyasını silin.")
        
    find_highest_poc_diff(cache_file)
