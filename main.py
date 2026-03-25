import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from scanner import read_tickers, calculate_poc

def filter_by_linear_regression(tickers, cache_file="bist_daily_1mo_cache.pkl"):
    print(f"Toplam {len(tickers)} hisse için 1 aylık günlük veri çekiliyor ve Lineer Regresyon modeli kuruluyor...")
    chunk_size = 150
    dfs = []
    
    if os.path.exists(cache_file):
        print(f"'{cache_file}' bulundu, günlük veriler buradan okunuyor. (Taze veri için dosyayı silin)")
        data = pd.read_pickle(cache_file)
    else:
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            print(f"Günlük Veri İndiriliyor: Grup {i//chunk_size + 1} / {(len(tickers) // chunk_size) + 1}...")
            try:
                # pandas warningleri önlemek için progress=False
                chunk_data = yf.download(chunk, period="1mo", interval="1d", progress=False)
                if not chunk_data.empty:
                    if isinstance(chunk_data.columns, pd.MultiIndex):
                        cols_to_keep = [col for col in chunk_data.columns if col[0] == 'Close']
                        dfs.append(chunk_data[cols_to_keep])
            except Exception:
                pass
            time.sleep(0.5)
            
        print("Günlük veri birleştiriliyor...")
        if dfs:
            data = pd.concat(dfs, axis=1)
            data.to_pickle(cache_file)
            print(f"Günlük veriler '{cache_file}' dosyasına kaydedildi!")
        else:
            print("Günlük veri indirme başarısız.")
            return []

    passed_tickers = []
    
    try:
        df_tickers = sorted(list(set(data.columns.get_level_values(1))))
    except:
        print("Beklenmeyen günlük veri yapısı.")
        return []
        
    for ticker in df_tickers:
        try:
            if ('Close', ticker) in data.columns:
                series = data['Close'][ticker].dropna()
                if len(series) < 10:  # En az 10 günlük veri olsun
                    continue
                    
                y = series.values
                x = np.arange(len(y))
                
                slope, intercept = np.polyfit(x, y, 1)
                
                y_pred = slope * x + intercept
                y_mean = np.mean(y)
                ss_tot = np.sum((y - y_mean)**2)
                
                if ss_tot == 0:
                    continue
                    
                ss_res = np.sum((y - y_pred)**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                if slope > 5 and r_squared >= 0.50:
                    passed_tickers.append(ticker)
        except Exception:
            pass
            
    print(f"R^2 >= %75 ve pozitif eğime sahip hisse sayısı ({len(tickers)} üzerinden): {len(passed_tickers)}")
    return passed_tickers

def run_backtest(all_tickers, filtered_tickers, cache_file="bist_all_backtest_cache.pkl"):
    print(f"Toplam {len(all_tickers)} hisse için son 5 günlük 1 dakikalık veri indirme/okuma kontrolü yapılıyor...")
    chunk_size = 150
    dfs = []
    
    if os.path.exists(cache_file):
        print(f"'{cache_file}' bulundu, tüm hisselerin 5 günlük verileri hızlıca buradan okunuyor.")
        data = pd.read_pickle(cache_file)
    else:
        print(f"Önbellek bulunamadı. Toplam {len(all_tickers)} hisse için veri indirilecek. Bu işlem sadece bir kez yapılacak!")
        for i in range(0, len(all_tickers), chunk_size):
            chunk = all_tickers[i:i+chunk_size]
            print(f"İndiriliyor: Grup {i//chunk_size + 1} / {(len(all_tickers) // chunk_size) + 1}...")
            try:
                # `period="5d"` to cover yesterday and today safely
                chunk_data = yf.download(chunk, period="5d", interval="1m", progress=False)
                if not chunk_data.empty:
                    if isinstance(chunk_data.columns, pd.MultiIndex):
                        cols_to_keep = [col for col in chunk_data.columns if col[0] in ['Close', 'Volume']]
                        dfs.append(chunk_data[cols_to_keep])
                    else:
                        pass
            except Exception as e:
                print(f"Hata Group {i//chunk_size + 1}: {e}")
            time.sleep(0.5)
            
        print("Veri birleştiriliyor...")
        if dfs:
            data = pd.concat(dfs, axis=1)
            data.to_pickle(cache_file)
            print(f"Tüm veriler {cache_file} doysasına kaydedildi!")
        else:
            print("İndirme başarısız veya veri gelmedi.")
            return

    # Mümkün olan tarihleri çıkar
    dates = pd.Series(data.index).dt.normalize().unique()
    
    if len(dates) < 2:
        print("Yeterli gün (en az 2 gün) verisi yok!")
        return
        
    # Bugünü ve dünü belirle
    date_today = dates[-1]
    date_yesterday = dates[-2]
    
    print(f"\n--- ZAMAN DİLİMLERİ ---")
    print(f"Dünkü Veri Günü: {date_yesterday.strftime('%Y-%m-%d')}")
    print(f"Bugünkü Veri Günü: {date_today.strftime('%Y-%m-%d')}")
    print(f"-----------------------\n")
    
    yesterday_mask = data.index.normalize() == date_yesterday
    today_mask = data.index.normalize() == date_today
    
    df_yesterday = data.loc[yesterday_mask]
    df_today = data.loc[today_mask]
    
    try:
        # data df'sindeki multi-index'ten sembolleri çek
        all_cached_tickers = sorted(list(set(data.columns.get_level_values(1))))
        # Sadece parametre değiştirerek tekrar hesaplatabilmeniz için, filtreden geçenleri dahil ediyoruz:
        df_tickers = [t for t in all_cached_tickers if t in filtered_tickers]
    except:
        print("Beklenmeyen veri yapısı, sütunları kontrol edin.")
        return
        
    results = []
    
    print("Son 5 günlük tüm mumlar hesaba katılarak Güncel POC'ler hesaplanıyor...")
    for ticker in df_tickers:
        try:
            # 5 Günlük Serinin tamamı
            if ('Close', ticker) in data.columns and ('Volume', ticker) in data.columns:
                close_5d = data['Close'][ticker].dropna()
                vol_5d = data['Volume'][ticker].dropna()
                
                if close_5d.empty or vol_5d.empty:
                    continue
                    
                # Kar/Zarar hesabı için hala dünkü kapanış lazım
                if ('Close', ticker) in df_yesterday.columns:
                    close_yest_series = df_yesterday['Close'][ticker].dropna()
                    if close_yest_series.empty:
                        continue
                    last_price_yesterday = close_yest_series.iloc[-1]
                else:
                    continue
                
                # Bugünkü veriler
                if ('Close', ticker) in df_today.columns:
                    close_today_series = df_today['Close'][ticker].dropna()
                    if close_today_series.empty:
                        continue
                    last_price_today = close_today_series.iloc[-1]
                else:
                    continue
                
                poc_price, _ = calculate_poc(close_5d, vol_5d)
                
                if poc_price is not None and last_price_today > 0:
                    diff_pct = ((poc_price - last_price_today) / last_price_today) * 100
                    
                    # Bugüne kadar POC değerine geldi mi? 
                    # (Bugünkü en düşük fiyat ile en yüksek fiyat arasında veya yakınında mı)
                    today_min = close_today_series.min()
                    today_max = close_today_series.max()
                    
                    # Kücük bir tolerans payla bakalım (%0.5)
                    margin = poc_price * 0.005
                    if (today_min - margin) <= poc_price <= (today_max + margin):
                        touched = "EVET"
                    else:
                        touched = "HAYIR"
                    
                    # Sadece POC'si fiyatından yukarıda veya tam fiyatta olanları getir (Aşağıda kalanları hariç tut)
                    if diff_pct >= 0:
                        profit_loss = ((last_price_today - last_price_yesterday) / last_price_yesterday) * 100
                        
                        # Temas ettiğinde satılsaydı oluşacak kâr/zarar (%): (POC - Dün Kapanış) / Dün Kapanış
                        if touched == "EVET":
                            poc_profit_loss = ((poc_price - last_price_yesterday) / last_price_yesterday) * 100
                        else:
                            poc_profit_loss = 0.0
                            
                        results.append({
                            'Ticker': ticker,
                            'YestLast': last_price_yesterday,
                            'POC': poc_price,
                            'DiffPct': diff_pct,
                            'TodayLast': last_price_today,
                            'ProfitLoss': profit_loss,
                            'Touched': touched,
                            'PocProfitLoss': poc_profit_loss
                        })
        except Exception as e:
            continue
            
    if not results:
        print("Uygun hisse bulunamadı (Tüm hisselerin güncel fiyatı güncel POC'nin üzerinde veya hata oluştu).")
        return
        
    # Aradaki farkı en yüksek olanlara göre sırala 
    results.sort(key=lambda x: x['DiffPct'], reverse=True)
    top_20 = results[:20]
    
    print("\n" + "="*110)
    print("📈 GÜNCEL FİYATINA GÖRE 5 GÜNLÜK POC'Sİ EN YUKARIDA OLANLAR VE GÜNLÜK KAR/ZARARLARI")
    print("="*110)
    print(f"{'Hisse':<12} | {'Dün Kapanış':<12} | {'5G POC':<10} | {'Uzaklık Fark(%)':<15} | {'Bugün Son':<10} | {'Gün Sonu K/Z(%)':<15} | {'Temas':<6} | {'Temas K/Z(%)':<12}")
    print("-" * 110)
    
    total_pl = 0
    for res in top_20:
        pl_sign_str = f"%{res['ProfitLoss']:.2f}"
        poc_pl_str = f"%{res['PocProfitLoss']:.2f}" if res['Touched'] == "EVET" else "-"
        total_pl += res['ProfitLoss']
        
        print(f"{res['Ticker']:<12} | {res['YestLast']:<12.2f} | {res['POC']:<10.2f} | %{res['DiffPct']:<14.2f} | {res['TodayLast']:<10.2f} | {pl_sign_str:<15} | {res['Touched']:<6} | {poc_pl_str:<12}")
        
    print("-" * 110)
    if top_20:
        avg_pl = total_pl / len(top_20)
        print(f"\nTop 20 hissenin bugünkü ortalama getirisi: %{avg_pl:.2f}\n")
        
        try:
            df_results = pd.DataFrame(top_20)
            excel_filename = f"backtest_results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
            df_results.to_excel(excel_filename, index=False)
            print(f"Sonuçlar '{excel_filename}' dosyasına başarıyla kaydedildi!\n")
        except Exception as e:
            print(f"Excel'e kaydedilirken hata oluştu: {e}\n")

if __name__ == "__main__":
    tickers_file = "stox.txt"
    tickers = read_tickers(tickers_file)
    print(f"'{tickers_file}' içerisinden {len(tickers)} geçerli hisse çıkarıldı.")
    
    # 1. Aşama: 1 Aylık Günlük Lineer Regresyon Filtresi (R^2 >= 0.75 ve Eğim > 0)
    filtered_tickers = filter_by_linear_regression(tickers)
    
    # 2. Aşama: Olanlar üzerinden Backtest / POC (Hacim Profili) Stratejisini çalıştır
    if len(filtered_tickers) > 0:
        # Parametreleri (slope, R^2 vb) değiştirdiğinizde verileri baştan indirmeden 
        # hızlıca test edebilmek için veriyi tüm listeye göre bir kez çekip (cache'leyip), hesaplamayı filtrelilerde yapıyoruz.
        run_backtest(all_tickers=tickers, filtered_tickers=filtered_tickers, cache_file="bist_all_backtest_cache.pkl")
    else:
        print("Lineer Regresyon filtresini geçen hisse bulunamadı. Program sonlandırılıyor.")
