# Econometría Tarea 1
# Guillem Borrás, Johannes Felchner y Gonzalo Moll
# Funciones auxiliares al cuaderno de trabajo

#%%

# LIBRERÍAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from scipy.stats import norm, jarque_bera, t
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

#%%

# APARTADO 1
def simular_tamaño(T: int = 20, n_sims: int = 10000, 
                   alpha: float = 0.05, dist: str = "normal", df_t: int = 3, seed: int = 241) -> float:
    """
    Simula n_sims muestras de tamaño T de una distribución (normal o t-student)
    y calcula la proporción de rechazos del test de Jarque-Bera al nivel alpha.
    Devuelve el tamaño empírico (proporción de rechazos).
    """
    rng = np.random.default_rng(seed)
    rechazos = 0
    
    for i in range(n_sims):
        if dist == "normal":
            muestra = rng.normal(loc=0.0, scale=1.0, size=T)
        elif dist == "tstudent":
            try: 
                muestra = t.rvs(df_t, size=T, random_state=rng) 
            except TypeError: 
                muestra = rng.standard_t(df_t, size=T)
        else:
            raise ValueError("Distribución no reconocida: use 'normal' o 'tstudent'")
    
        jb_stat, p_value = jarque_bera(muestra)
        if p_value < alpha:
            rechazos += 1
    
    tamaño_empírico = rechazos / n_sims

    return rechazos, tamaño_empírico

#%%

# APARTADO 2
def pronosticar_y_evaluar_serie(nombre_activo, orden, seasonal_order, data_dict, tipo="precio", N=200, trend='n'):
    """
    Pronóstico rolling eficiente a un período hacia adelante sin volver a ajustar el modelo en cada paso.
    
    Parámetros:
    ----------
    nombre_activo : str
        Nombre del activo (clave en el diccionario).
    orden : tuple
        Orden del modelo ARIMA.
    seasonal_order : tuple
        Orden estacional del modelo SARIMA.
    data_dict : dict
        Diccionario con las series ('Rent_' o 'Prec_').
    tipo : str
        'precio' o 'rendimiento' para ajustar etiquetas y títulos.
    N : int
        Número de observaciones que se eliminarán de T.
    """

    print(f"Pronóstico rolling para {nombre_activo} ({tipo})")

    prefijo = "Prec" if tipo.lower().startswith("p") else "Rent"

    serie_train = data_dict[f"{prefijo}_train_{nombre_activo}"]
    serie_total = data_dict[f"{prefijo}_total_{nombre_activo}"]

    T_total = len(serie_total)
    indice_test_1 = T_total - N
    serie_test = serie_total[indice_test_1:]

    resumen, res = fit_model(
        serie_train,
        order=orden,
        seasonal_order=seasonal_order,
        trend=trend
    )

    preds, lower_ci, upper_ci = [], [], []

    for t in serie_test.index:
        forecast_res = res.get_forecast(steps=1)
        pred_mean = forecast_res.predicted_mean.iloc[0]
        conf_int = forecast_res.conf_int(alpha=0.05)
        ci_lower = conf_int.iloc[0, 0]
        ci_upper = conf_int.iloc[0, 1]

        preds.append(pred_mean)
        lower_ci.append(ci_lower)
        upper_ci.append(ci_upper)

        valor_real = serie_total.loc[t]
        try:
            res = res.extend(endog=[valor_real])
        except Exception:
            res = res.apply(endog=[valor_real], refit=False)

    preds = pd.Series(preds, index=serie_test.index, name="Pronóstico")
    lower_ci = pd.Series(lower_ci, index=serie_test.index, name="IC_inf")
    upper_ci = pd.Series(upper_ci, index=serie_test.index, name="IC_sup")

    mask = preds.notna() & serie_test.notna()
    rmse = sqrt(mean_squared_error(serie_test[mask], preds[mask]))
    mae = mean_absolute_error(serie_test[mask], preds[mask])

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

    if tipo.lower().startswith("p"):
        label_real = "Precios reales"
        label_pred = "Precios pronosticados"
        ylabel = "Precio"
    else:
        label_real = "Rentabilidades reales"
        label_pred = "Rentabilidades pronosticadas"
        ylabel = "Rentabilidad"

    plt.figure(figsize=(12, 6))
    plt.plot(serie_test.index, serie_test, color='black', lw=1.5, label=label_real)
    plt.plot(preds.index, preds, color='royalblue', lw=2, label=label_pred)
    plt.fill_between(preds.index, lower_ci, upper_ci, color='deepskyblue', alpha=0.25, label='IC 95%')
    plt.title(f"Pronóstico rolling – ({nombre_activo})", fontsize=14, fontweight='bold')
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='upper right', fontsize=10, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    return {
        "Activo": nombre_activo,
        "Tipo": tipo,
        "RMSE": rmse,
        "MAE": mae,
    }

#%%

# APARTADO 3
def plot_time_series(prices,
                base: float = 100.0,
                log_scale: bool = False,
                figsize: tuple[int, int] = (11,6),
                grid: bool = True,
                legend: bool = True,
                save_path: str | None = None):
    """Dibuja las series de precios de un DataFrame/Series de pandas."""
    pr = prices.copy()

    # Asegurar numérico y limpiar filas completamente vacías
    pr = pr.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if pr.empty:
        raise ValueError("No hay datos numéricos válidos para representar.")

    # Usamos índice temporal para el eje "x" en el gráfico
    if not pd.api.types.is_datetime64_any_dtype(pr.index):
        try:
            pr.index = pd.to_datetime(pr.index, errors='raise')
        except Exception:
            pass
    pr = pr.sort_index()

    # Crear figura y dibujar
    fig, ax = plt.subplots(figsize=figsize)
    pr.plot(ax=ax)

    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    if grid:
        # Grid tanto vertical como horizontal
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.4)
    if legend:
        ax.legend(loc='best', frameon=False)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def fit_model(
    y,
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 0),
    trend='n',
    exog=None,
    enforce_stationarity=True,
    enforce_invertibility=True
):
    """
    Ajusta ARMA/ARIMA/SARIMA (con o sin exógenas) y retorna tabla de parámetros + residuos.
    
    Parameters
    ----------
    y : pd.Series
        Serie temporal.
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s)
    trend : {'n', 'c', 't', 'ct'}
    exog : array-like, pd.Series o pd.DataFrame, optional
        Variable(s) exógena(s) (por ejemplo, una dummy escalón).
    enforce_stationarity : bool
        Si True, fuerza estacionariedad.
    enforce_invertibility : bool
        Si True, fuerza invertibilidad.
    
    Returns
    -------
    tuple (pd.DataFrame, SARIMAXResults)
        - DataFrame: tabla con ['param', 'coef', 'stderr', 'z', 'pvalue', 'sig']
          (excluye sigma2)
        - Objeto de resultados SARIMAXResults (incluye residuos, predicciones, etc.)
    """
    # Inferir frecuencia si es DatetimeIndex
    y = y.copy()
    if isinstance(y.index, pd.DatetimeIndex):
        freq = pd.infer_freq(y.index) or 'B'
        y = y.asfreq(freq)
    
    # Ajuste del modelo
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        exog=exog,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    )
    res = model.fit(disp=False)
    
    # Extraer tabla de parámetros
    summary_df = pd.DataFrame({
        'param': res.params.index,
        'coef': res.params.values,
        'stderr': res.bse.values,
        'z': res.tvalues.values,
        'pvalue': res.pvalues.values
    })
    
    # Eliminar sigma2 de la tabla
    summary_df = summary_df[summary_df['param'] != 'sigma2'].reset_index(drop=True)
    
    # Añadir nivel de significatividad
    summary_df['sig'] = pd.cut(
        summary_df['pvalue'],
        bins=[-np.inf, 0.01, 0.05, 0.1, np.inf],
        labels=['***', '**', '*', '']
    )
    
    # Guardar el resultado completo en los atributos del DataFrame
    summary_df.attrs['results'] = res
    
    return summary_df, res

def mean_variance_plot(series, window=50):
    """
    Calcula medias y varianzas por bloques y dibuja el gráfico media-varianza.
    
    Parameters
    ----------
    series : pd.Series
        Serie temporal (con índice de tiempo).
    window : int
        Tamaño del bloque/ventana para calcular media y varianza.
    """
    # Dividir la serie en bloques de tamaño 'window'
    means, vars_ = [], []
    for i in range(0, len(series), window):
        block = series.iloc[i:i+window].dropna()
        if len(block) > 1:
            means.append(block.mean()/1000) #divido por mil para que la escala no sea tan grande
            vars_.append(block.var()/1000)

    # Hacer gráfico
    plt.figure(figsize=(11,6))
    plt.scatter(means, vars_, alpha=0.7, edgecolors='k')
    plt.xlabel(r"$ \mu_{IBEX,60} \cdot 10^{-3}$")
    plt.ylabel(r"$ \sigma^2_{IBEX,60} \cdot 10^{-3}$")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

def ACF_PACF(res_or_residuals, lags: int = 40, k: int = 1, plot: bool = True):
    try:
        e = pd.Series(np.asarray(res_or_residuals.resid)).dropna()
    except AttributeError:
        e = pd.Series(np.asarray(res_or_residuals)).dropna()

    nlags = min(lags, len(e) - 1)
    if nlags < 1: raise ValueError("No hay suficientes datos.")
    k = int(max(0, min(k, nlags)))

    acf_full = sm_acf(e, nlags=nlags, fft=True)
    pacf_full = sm_pacf(e, nlags=nlags, method="ywmle")
    lags_range = np.arange(k, nlags + 1)
    acf_vals, pacf_vals = acf_full[k:], pacf_full[k:]

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plot_acf(e, lags=lags_range, ax=ax1, zero=False)
        plot_pacf(e, lags=lags_range, ax=ax2, method="ywmle", zero=False)
        ax1.set_title("ACF "); ax2.set_title("PACF ")
        ax1.grid(False); ax2.grid(False)

        # Ticks x mínimos y desde k
        def _ticks(ax):
            if nlags <= k: ax.set_xticks([k]); ax.set_xlim(k-0.5, k+0.5); return
            step = max(1, (nlags - k) // 8)
            ticks = np.arange(k, nlags + 1, step)
            if ticks[0] != k: ticks = np.insert(ticks, 0, k)
            if ticks[-1] != nlags: ticks = np.append(ticks, nlags)
            ax.set_xticks(ticks); ax.set_xlim(k - 0.5, nlags + 0.5)
        _ticks(ax1); _ticks(ax2)
        
        # Banda de confianza ±1.96/sqrt(N)
        conf = 1.96 / np.sqrt(len(e))
        # Reajuste del eje y desde k, incluyendo margen para que se vean los extremos de la banda de confianza        
        m1 = max(conf, float(np.nanmax(np.abs(acf_vals))) if acf_vals.size else conf)
        m2 = max(conf, float(np.nanmax(np.abs(pacf_vals))) if pacf_vals.size else conf)
        pad = 1.10
        ax1.set_ylim(-pad*m1, pad*m1)
        ax2.set_ylim(-pad*m2, pad*m2)

        plt.tight_layout(); plt.show()

    return {"residuals": e.values, "acf": acf_vals, "pacf": pacf_vals, "lags": lags_range}

def diebold_mariano_test(
    y,
    model1_order,
    model2_order,
    seasonal_order1=(0, 0, 0, 0),
    seasonal_order2=(0, 0, 0, 0),
    h=1,
    K=20,
    trend='n',
    enforce_stationarity=True,
    enforce_invertibility=True,
    loss="MAE"   # "MAE" (abs) o "MSE"
):
    """
    Test de Diebold-Mariano para comparar capacidad predictiva de dos modelos ARIMA/SARIMA.

    Parámetros
    ----------
    y : pd.Series o array-like
        Serie temporal observada.
    model1_order : tuple (p, d, q)
        Orden no estacional del primer modelo.
    model2_order : tuple (p, d, q)
        Orden no estacional del segundo modelo.
    seasonal_order1 : tuple (P, D, Q, s), default (0,0,0,0)
        Orden estacional del primer modelo (SARIMA). Si s=0 ó P=D=Q=0, no hay estacionalidad.
    seasonal_order2 : tuple (P, D, Q, s), default (0,0,0,0)
        Orden estacional del segundo modelo (SARIMA).
    h : int, default 1
        Horizonte de predicción (h-step ahead).
    K : int, default 20
        Nº de predicciones dentro de muestra (ventana rolling sobre las últimas K observaciones).
    trend : {'n','c','t','ct'}, default 'n'
        Tendencia del modelo SARIMAX.
    enforce_stationarity, enforce_invertibility : bool
        Restricciones en el ajuste.
    loss : {"MAE","MSE"}, default "MAE"
        Función de pérdida para comparar (errores absolutos o cuadrados).

    Returns
    -------
    dict con claves:
        - 'DM': estadístico Diebold–Mariano (aprox. N(0,1))
        - 'p_value': p-valor bilateral
        - 'dbar': media de diferencias de pérdidas
        - 'var_dbar': varianza Newey–West de la media de diferencias
        - 'interpretation': texto con la conclusión
        - 'model1_order', 'seasonal_order1', 'model2_order', 'seasonal_order2'
    
    Notas
    -----
    H0: Ambos modelos tienen igual capacidad predictiva.
    H1: Capacidades predictivas distintas.

    Signo del DM (si se rechaza H0):
        - DM < 0  -> Modelo 1 mejor (menor pérdida media)
        - DM > 0  -> Modelo 2 mejor
    """
    # A array 1D
    y = np.asarray(y).flatten()
    T = len(y)

    if K <= 0 or K >= T:
        raise ValueError("K debe ser > 0 y < T.")

    # Contenedores para predicciones (solo rellenaremos las últimas K posiciones)
    y1_f = np.zeros(T)
    y2_f = np.zeros(T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for k in range(1, K + 1):
            # Índice de la observación a predecir (dentro de las últimas K)
            idx_pred = T - K + k - 1

            # Conjunto de entrenamiento hasta el instante anterior al objetivo (para h=1)
            # Para h>1, entrenamos hasta el instante que deja h pasos al objetivo
            y_train = y[:T - K + k - h]

            # --- Modelo 1 (SARIMA) ---
            m1 = SARIMAX(
                y_train,
                order=model1_order,
                seasonal_order=seasonal_order1,
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            ).fit(disp=False)

            f1 = m1.get_forecast(steps=h).predicted_mean
            y1_f[idx_pred] = np.asarray(f1).flatten()[-1]

            # --- Modelo 2 (SARIMA) ---
            m2 = SARIMAX(
                y_train,
                order=model2_order,
                seasonal_order=seasonal_order2,
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            ).fit(disp=False)

            f2 = m2.get_forecast(steps=h).predicted_mean
            y2_f[idx_pred] = np.asarray(f2).flatten()[-1]

    # Errores sobre las K últimas observaciones
    y_test = y[T - K:]
    e1 = y_test - y1_f[T - K:]
    e2 = y_test - y2_f[T - K:]

    d = np.abs(e1) - np.abs(e2)
    dbar = float(np.mean(d))

    # Varianza Newey–West de la media de d (autocorrelación hasta N_cov)
    gamma0 = np.var(d, ddof=1)
    N_cov = int(np.round(K ** (1/3) - 0.5)) + 1  # misma regla que en código MATLAB

    # ACF y autovarianzas hasta N_cov rezagos (sin el lag 0)
    acf_vals = sm_acf(d, nlags=N_cov, fft=False)
    vgammak = acf_vals[1:N_cov + 1] * gamma0   # incluye lag = N_cov

    # Versión sin pesos (manteniendo  criterio MATLAB):
    var_dbar = (1 / K) * (gamma0 + 2 * np.sum(vgammak))

    # Estadístico y p-valor
    DM = dbar / np.sqrt(var_dbar) if var_dbar > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(np.abs(DM))) if np.isfinite(DM) else np.nan

    # Interpretación
    if np.isnan(p_value):
        interp = "No se pudo calcular el estadístico de forma estable (varianza ~ 0)."
    elif p_value < 0.05:
        if DM < 0:
            interp = (f"Modelo 1 (ARIMA{model1_order}×SARIMA{seasonal_order1}) "
                      f"significativamente MEJOR (p={p_value:.4f}).")
        else:
            interp = (f"Modelo 2 (ARIMA{model2_order}×SARIMA{seasonal_order2}) "
                      f"significativamente MEJOR (p={p_value:.4f}).")
    else:
        interp = f"No hay diferencia significativa entre modelos (p={p_value:.4f})."

    return {
        'DM': float(DM),
        'p_value': float(p_value),
        'dbar': float(dbar),
        'var_dbar': float(var_dbar),
        'interpretation': interp,
        'model1_order': model1_order,
        'seasonal_order1': seasonal_order1,
        'model2_order': model2_order,
        'seasonal_order2': seasonal_order2,
        'loss': loss.upper()
    }

def _horizon_and_index(last_date: pd.Timestamp, calendar: str):
    last_date = pd.Timestamp(last_date)
    year = last_date.year
    tz = getattr(last_date, "tz", None)
    if calendar == "business":
        start = last_date + pd.Timedelta(days=1)
        end = pd.Timestamp(year=year, month=12, day=31, tz=tz)
        idx = pd.bdate_range(start, end, freq="B")
    elif calendar == "daily":
        start = last_date + pd.Timedelta(days=1)
        end = pd.Timestamp(year=year, month=12, day=31, tz=tz)
        idx = pd.date_range(start, end, freq="D")
    elif calendar == "monthly":
        # meses completos restantes hasta diciembre (periodos de fin de mes)
        cur = pd.Period(last_date, freq="M")
        end = pd.Period(f"{year}-12", freq="M")
        if cur == end:
            idx = pd.PeriodIndex([], freq="M").to_timestamp("M")
        else:
            idx = pd.period_range(cur+1, end, freq="M").to_timestamp("M")
    elif calendar == "quarterly":
        # trimestres restantes (periodos de fin de trimestre)
        cur = pd.Period(last_date, freq="Q")
        end = pd.Period(f"{year}Q4", freq="Q")
        if cur == end:
            idx = pd.PeriodIndex([], freq="Q").to_timestamp("Q")
        else:
            idx = pd.period_range(cur+1, end, freq="Q").to_timestamp("Q")
    else:
        raise ValueError("calendar debe ser 'business', 'daily', 'monthly' o 'quarterly'")
    return len(idx), pd.DatetimeIndex(idx)

def simulate_and_eval(
    res,
    current_price: float | None,
    last_date: pd.Timestamp,
    calendar: str = "business",       # 'business' | 'daily' | 'monthly' | 'quarterly'
    N: int = 20_000,
    random_state: int | None = None,
    prices_context: pd.Series | None = None,
    n_past: int = 60,
    plot: bool = True,
    n_paths_plot: int = 200,
    start_at_t: bool = True,
    return_paths: bool = False,
    to_level = np.exp,                 # None si el modelo ya está en niveles
):
    """
    Simula hasta fin de año respetando la granularidad temporal vía `calendar`.
    """

    if random_state is not None:
        np.random.seed(int(random_state))

    H, future_index = _horizon_and_index(last_date, calendar)
    if H == 0:
        if plot:
            fig, ax = plt.subplots(figsize=(12, 6))
            if prices_context is not None and len(prices_context) > 0:
                P_hist = prices_context.tail(n_past)
                ax.plot(P_hist.index, P_hist.values, label=f"Histórico (últimos {n_past})")
            if current_price is not None:
                ax.axhline(current_price, linestyle=":", linewidth=1.5, label=f"Umbral = {current_price:.2f}")
            ax.set_xlabel("Fecha"); ax.set_ylabel("Nivel simulado")
            ax.legend(); ax.grid(True); plt.tight_layout(); plt.show()
        return {"prob_gain": 0.0, "H": 0, "P_t": None if current_price is None else float(current_price),
                "note": "No quedan periodos hasta fin de año."}

    try:
        sim = res.simulate(steps=H, repetitions=N, anchor="end")
    except TypeError:
        sim = res.simulate(nsimulations=H, repetitions=N, anchor="end")

    X = np.asarray(getattr(sim, "to_numpy", lambda: sim)())  # shape (H, N)

    # Convertir a niveles si procede
    P_future = to_level(X) if callable(to_level) else X

    if start_at_t and (current_price is not None):
        P0 = np.full((1, N), float(current_price))
        P_paths = np.vstack([P0, P_future])      # (H+1, N)
        t_index = pd.DatetimeIndex([last_date]).append(future_index)
    else:
        P_paths = P_future
        t_index = future_index

    # Estadísticos
    mean_path = P_paths.mean(axis=1)
    p05_path  = np.percentile(P_paths,  5, axis=1)
    p50_path  = np.percentile(P_paths, 50, axis=1)
    p95_path  = np.percentile(P_paths, 95, axis=1)

    prob_gain = None
    if current_price is not None:
        prob_gain = float((P_paths[-1, :] > current_price).mean())

    # Gráfico
    if plot:
        fig, ax = plt.subplots(figsize=(12, 6))

        if prices_context is not None and len(prices_context) > 0:
            P_hist = prices_context.tail(n_past)
            ax.plot(P_hist.index, P_hist.values, label=f"Histórico (últimos {n_past})")

        m = int(min(n_paths_plot, P_paths.shape[1]))
        if m > 0:
            cols = np.linspace(0, P_paths.shape[1] - 1, m, dtype=int)
            ax.plot(t_index, P_paths[:, cols], alpha=0.15, linewidth=0.8)

        ax.plot(t_index, mean_path, linewidth=2, label="Media simulada")
        ax.plot(t_index, p50_path, linestyle="--", linewidth=1.3, label="P50 (mediana)")
        ax.plot(t_index, p05_path, linestyle="--", linewidth=1.0, label="P5")
        ax.plot(t_index, p95_path, linestyle="--", linewidth=1.0, label="P95")

        if current_price is not None:
            ax.axhline(current_price, linestyle=":", linewidth=1.5, label=f"Umbral = {current_price:.2f}")

        ax.set_xlabel("Fecha")
        ax.set_ylabel("Nivel simulado")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    out = {
        "prob_gain": prob_gain,
        "H": H,
        "future_index": t_index,
        "P_t": None if current_price is None else float(current_price),
        "mean_T": float(mean_path[-1]),
        "p50_T":  float(p50_path[-1]),
        "p05_T":  float(p05_path[-1]),
        "p95_T":  float(p95_path[-1]),
        "N": N,
        "calendar": calendar,
        "n_past": n_past,
        "start_at_t": start_at_t
    }
    if return_paths:
        out["paths_prices"] = P_paths
    return out