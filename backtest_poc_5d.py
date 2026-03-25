import pandas as pd
import numpy as np
import os
import time
from scanner import read_tickers, calculate_poc

def run_5d_buy_sell_backtest(cache_file="bist_all_backtest_cache.pkl", daily_cache="bist_daily_1mo_cache.pkl"):
    print(f"'{cache_file}' dosyasından veriler okunuyor...")
    if not os.path.exists(cache_file):
        print("Hata: Önbellek dosyası bulunamadı. Lütfen önce main.py çalıştırarak verileri indirin.")
        return
        
    data = pd.read_pickle(cache_file)
    
    # Mümkün olan günleri bul (Normalde son 5 iş günü olmalı)
    dates = pd.Series(data.index).dt.normalize().unique()
    if len(dates) < 2:
        print("Yeterli gün verisi yok!")
        return
        
    day1_date = dates[0]   # 5 gün önceki ilk gün
    last_date = dates[-1]  # Bugün (Son gün)
    
    print(f"Başlangıç (Alım) Günü: {day1_date.strftime('%Y-%m-%d')}")
    print(f"Bitiş Günü (Bugün): {last_date.strftime('%Y-%m-%d')}")
    
    day1_mask = data.index.normalize() == day1_date
    future_mask = data.index.normalize() > day1_date # Alımdan sonraki günler (Satış takip edilecek)
    
    df_day1 = data.loc[day1_mask]
    df_future = data.loc[future_mask]
    
    # 1 AYLIK FİLTRELEME
    valid_monthly_tickers = set()
    if os.path.exists(daily_cache):
        print(f"\nAylık (%30+ getiri) filtresi için '{daily_cache}' inceleniyor...")
        try:
            daily_data = pd.read_pickle(daily_cache)
            tickers_monthly = list(set(daily_data.columns.get_level_values(1)))
            for t in tickers_monthly:
                if ('Close', t) in daily_data.columns:
                    s = daily_data['Close'][t].dropna()
                    if len(s) > 1:
                        first_price = s.iloc[0]
                        last_price = s.iloc[-1]
                        if first_price > 0:
                            perf = ((last_price - first_price) / first_price) * 100
                            if perf > 30:
                                valid_monthly_tickers.add(t)
            print(f"Borsada 1 aylık performansı %30'dan büyük olan hisse sayısı: {len(valid_monthly_tickers)}")
        except Exception as e:
            print(f"Aylık veri okunurken hata oluştu: {e}")
    else:
        print(f"Uyarı: Aylık veriler ({daily_cache}) bulunamadı. Lütfen main.py'i çalıştırın! (Filtre uygulanmıyor)")
    
    try:
        all_tickers = sorted(list(set(data.columns.get_level_values(1))))
        if valid_monthly_tickers:
            tickers = [t for t in all_tickers if t in valid_monthly_tickers]
            print(f"Hem 5 günlük verisi olan hem de %30+ getiri filtresinden geçen hisse sayısı: {len(tickers)}")
        else:
            tickers = all_tickers
    except:
        print("Beklenmeyen veri yapısı, sütunları kontrol edin.")
        return
        
    results = []
    
    print("\n5 Gün önceki (1. Gün) duruma göre alımlar yapılıyor ve takip ediliyor...")
    for ticker in tickers:
        try:
            if ('Close', ticker) in df_day1.columns and ('Volume', ticker) in df_day1.columns:
                close_day1 = df_day1['Close'][ticker].dropna()
                vol_day1 = df_day1['Volume'][ticker].dropna()
                
                if close_day1.empty or vol_day1.empty:
                    continue
                    
                # 1. Günün POC'si + Volume Density hesabı
                poc_price, _ = calculate_poc(close_day1, vol_day1)
                
                # 1. Gün sonundaki kapanış fiyatı
                buy_price = close_day1.iloc[-1]
                
                if poc_price is None or buy_price <= 0:
                    continue
                
                # POC'nin bulunduğu bin'deki hacmin toplam hacme oranını hesapla
                bins = 50
                valid_data = pd.concat([close_day1, vol_day1], axis=1).dropna()
                if valid_data.empty:
                    continue
                valid_close = valid_data.iloc[:, 0]
                valid_volume = valid_data.iloc[:, 1]
                price_bins = np.linspace(valid_close.min(), valid_close.max(), bins)
                indices = np.digitize(valid_close, price_bins)
                volume_profile = np.zeros(bins)
                for i in range(len(valid_close)):
                    idx = indices[i] - 1
                    if idx >= bins: idx = bins - 1
                    elif idx < 0: idx = 0
                    volume_profile[idx] += float(valid_volume.iloc[i])
                total_volume = volume_profile.sum()
                poc_index = np.argmax(volume_profile)
                poc_density = volume_profile[poc_index] / total_volume if total_volume > 0 else 0
                
                # DENSITY FİLTRESİ: POC noktasındaki hacim yoğunluğu en az %50 olmalı
                if poc_density < 0.10:
                    continue
                    
                # ALIM KRİTERİ: 1. Gün sonu fiyatı, 1. Günün POC'sinden AZ olmalı
                if buy_price < poc_price:
                    
                    if ('Close', ticker) not in df_future.columns:
                        continue
                        
                    future_close = df_future['Close'][ticker].dropna()
                    if future_close.empty:
                        continue
                    
                    reached = False
                    sell_price = 0.0
                    
                    # Gelecekteki fiyatlarda hedef (POC) fiyatına ulaşıp ulaşmadığını kontrol et
                    for price in future_close:
                        if price >= poc_price:
                            reached = True
                            sell_price = poc_price # O an hedeften (POC'den) satıldı varsayıyoruz
                            break
                            
                    if reached:
                        status = "Satabildi (POC'ye Geldi)"
                    else:
                        status = "Satamadı (Son Gün Kapanışından Satıldı)"
                        sell_price = future_close.iloc[-1] # Vade sonuna kadar hedefe gitmedi, bugün son fiyattan satıldı
                        
                    profit_loss_val = sell_price - buy_price
                    profit_loss_pct = (profit_loss_val / buy_price) * 100
                    
                    results.append({
                        'Hisse': ticker,
                        'Durum': status,
                        'Alış Fiyatı (1. Gün)': buy_price,
                        'Hedef POC (1. Gün)': poc_price,
                        'Satış Fiyatı': sell_price,
                        'Kar/Zarar (TL)': profit_loss_val,
                        'Kar/Zarar (%)': profit_loss_pct
                    })
        except Exception as e:
            continue
            
    if not results:
        print("Kriterlere uyan hiçbir hisse bulunamadı.")
        return
        
    # Sonuçları DataFrame'e çevirip Excel'e kaydetme
    df_results = pd.DataFrame(results)
    
    excel_filename = f"yeni_strateji_5d_poc_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    try:
        df_results.to_excel(excel_filename, index=False)
        print(f"\nİşlem Başarılı! Sonuçlar '{excel_filename}' adlı Excel dosyasına kaydedildi.")
    except Exception as e:
        print(f"\nExcel'e kaydedilirken hata oluştu: {e}")
    
    # Terminale basit bir özet bas
    sold_df = df_results[df_results['Durum'].str.contains("Satabildi")]
    fail_df = df_results[df_results['Durum'].str.contains("Satamadı")]
    
    total_profit = df_results['Kar/Zarar (TL)'].sum()
    avg_profit_pct = df_results['Kar/Zarar (%)'].mean()
    
    print(f"\n--- YENİ STRATEJİ ÖZETİ ---")
    print(f"Toplam Alınan Hisse Sayısı: {len(df_results)}")
    
    print(f"\n✅ Hedefe (POC) Gidip Kar ile 'Satılabilen' Hisseler ({len(sold_df)} adet):")
    if not sold_df.empty:
        print(", ".join(sold_df['Hisse'].tolist()))
    else:
        print("- Yok -")
        
    print(f"\n⏳ Hedefe Gitmeyip Bugün Sonundan 'Satılamayan/Düz Satılan' Hisseler ({len(fail_df)} adet):")
    if not fail_df.empty:
        print(", ".join(fail_df['Hisse'].tolist()))
    else:
        print("- Yok -")
        
    print(f"\n💵 Tüm Karların/Zararların Mutlak Toplamı (Sadece 1'er Lot İçin): {total_profit:.2f} ₺")
    print(f"📈 Ortalama Kar/Zarar Yüzdesi: %{avg_profit_pct:.2f}")
    
if __name__ == "__main__":
    run_5d_buy_sell_backtest()
