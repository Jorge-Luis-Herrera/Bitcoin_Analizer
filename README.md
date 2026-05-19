# Bitcoin Forecast V4 — Predicción de Precios con Modelos Fundacionales de Series Temporales

*Autor:* Jorge Luis Herrera Cecilia  
**Estado:** Producción  
**Última actualización:** 18 de Mayo del 2026

---

## Resumen

Este proyecto implementa un sistema de predicción del precio de Bitcoin (BTC-USD) basado en **modelos fundacionales de series temporales** (*Time Series Foundation Models*, TSFM). Se emplea **Amazon Chronos T5-Tiny** como núcleo del sistema, un modelo Transformer preentrenado en un corpus masivo de datos de series temporales de diversos dominios, capaz de realizar pronósticos en *zero-shot* —sin necesidad de ajuste fino— sobre datos financieros.

El sistema ofrece dos modalidades de predicción:

1. **Backtest (validación histórica):** Predice los últimos *N* días hacia atrás, utilizando exclusivamente la información disponible hasta cada punto de predicción, y compara el resultado contra el valor real. Proporciona métricas objetivas de desempeño (MAPE, RMSE, MAE).
2. **Forecast (pronóstico hacia adelante):** Predice los próximos *N* días utilizando todo el historial disponible, con intervalos de confianza probabilísticos (P10–P90).

Adicionalmente, incluye un **monitor en tiempo real** que actualiza datos de mercado cada 3 segundos y superpone el pronóstico actual sobre velas en vivo.

---

## 1. Marco Teórico

### 1.1 Predicción de Series Temporales Financieras

Las series de precios de criptoactivos presentan propiedades estadísticas que las hacen particularmente desafiantes para el modelado predictivo: alta volatilidad, heterocedasticidad, colas pesadas, y ausencia de estacionalidad clara. Tradicionalmente, los enfoques empleados incluyen modelos ARIMA/GARCH (Box & Jenkins, 1976), suavizado exponencial (Holt-Winters), y más recientemente, redes neuronales recurrentes (LSTM, GRU) y Transformers (Vaswani et al., 2017).

### 1.2 Modelos Fundacionales de Series Temporales (TSFM)

Inspirados por el éxito de los *Large Language Models* (LLMs) en NLP, los TSFMs se preentrenan en colecciones masivas y diversas de datos temporales para aprender patrones universales de dinámica temporal. A diferencia de los modelos entrenados *ad-hoc* para cada dominio, los TSFMs pueden ser empleados en *zero-shot* —es decir, sin entrenamiento adicional— sobre series nunca antes vistas.

**Amazon Chronos** (Ansari et al., 2024) pertenece a esta familia. Su arquitectura se basa en un codificador-decodificador T5 (Raffel et al., 2020) que opera sobre *parches* de la serie temporal (*patching*), una técnica que consiste en dividir la secuencia de entrada en bloques contiguos para reducir la dimensionalidad y capturar patrones locales. El modelo se preentrena con una función de pérdida de verosimilitud cuantílica, lo que le permite generar pronósticos probabilísticos.

### 1.3 Variantes de Chronos Evaluadas

| Variante | Parámetros | Arquitectura | MAPE (backtest) |
|----------|-----------|-------------|:---------------:|
| **Chronos-T5-Tiny** | ~8M | T5 encoder-decoder | **2.07%** |
| Chronos-T5-Small | ~46M | T5 encoder-decoder | 2.26% |
| Chronos-T5-Base | ~200M | T5 encoder-decoder | 2.16% |

La variante **Tiny** ofrece el mejor equilibrio entre precisión (MAPE 2.07%) y velocidad de inferencia (~0.15s por predicción), superando incluso a sus contrapartes más grandes en este dominio específico.

### 1.4 Métricas de Evaluación

- **MAPE** (*Mean Absolute Percentage Error*): \(\frac{1}{n}\sum_{i=1}^{n}\frac{|\hat{y}_i - y_i|}{y_i} \times 100\)
- **RMSE** (*Root Mean Square Error*): \(\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}\)
- **MAE** (*Mean Absolute Error*): \(\frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i|\)

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     ENTRADA (Input Layer)                    │
│  Yahoo Finance → BTC-USD (precio histórico, volumen)        │
│  CoinGecko   → Fear & Greed Index                          │
│  Yahoo Finance → S&P 500, Gold, DXY (contexto macro)       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│               CAPA DE PREDICCIÓN (Chronos)                   │
│  - Carga del modelo T5 preentrenado                          │
│  - Tokenización mediante parches (patching)                  │
│  - Inferencia autoregresiva con 20-100 muestras              │
│  - Agregación por cuantiles (mediana, P10, P90)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              CAPA DE VISUALIZACIÓN (Streamlit + Plotly)       │
│  - Dashboard interactivo con 4 pestañas                      │
│  - Gráficos dinámicos zoom-eables                           │
│  - Tablas de predicción con métricas                         │
│  - Monitor en tiempo real (actualización cada 3s)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Estructura del Proyecto

```
Bitcoin_Analizer/
├── versions/
│   └── hybrid_v4/              # CÓDIGO FUENTE PRINCIPAL
│       ├── app_v4.py           # Dashboard Streamlit (interfaz web)
│       ├── backtest_v4.py      # Backtest CLI (línea de comandos)
│       ├── forecast_v4.py      # Forecast CLI (línea de comandos)
│       └── estudio_backtest.py # Estudio multi-modelo
├── cache/                      # Datos descargados cacheados
├── docs/
│   └── parametros_impacto.md   # Documentación de parámetros de mercado
├── results/                    # Resultados generados
│   ├── backtest_v4.png         # Gráfica de backtest
│   ├── backtest_v4.csv         # Datos de backtest
│   ├── forecast_v4.png         # Gráfica de forecast
│   ├── forecast_v4.csv         # Datos de forecast
│   └── estudio_backtest_modelos.csv  # Comparativa multi-modelo
├── venv_py310/                 # Entorno virtual Python 3.10
├── .gitignore
└── README.md
```

---

## 4. Requisitos del Sistema

### 4.1 Dependencias de Software

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0 (backend de Chronos)
- Paquetes Python (instalados automáticamente):

| Paquete | Propósito |
|---------|-----------|
| `chronos-forecasting` | Modelo fundacional Chronos |
| `streamlit` | Dashboard web interactivo |
| `plotly` | Gráficos interactivos |
| `streamlit-autorefresh` | Auto-actualización en tiempo real |
| `yfinance` | Descarga de datos de mercado |
| `pandas`, `numpy` | Procesamiento de datos |
| `torch` | Backend de deep learning |
| `requests` | Consultas HTTP (Fear & Greed) |
| `scikit-learn` | Métricas de evaluación |

### 4.2 Hardware

- **CPU:** Cualquier procesador moderno (inferencia en CPU)
- **RAM:** ≥ 8 GB recomendado
- **Disco:** ~2 GB para el modelo y datos

---

## 5. Instalación y Uso Local

### 5.1 Instalación

```bash
# 1. Clonar o copiar el proyecto
cd Bitcoin_Analizer

# 2. Crear entorno virtual (Python 3.10+)
python3.10 -m venv venv_py310

# 3. Activar entorno
source venv_py310/bin/activate  # Linux/Mac
# o: venv_py310\Scripts\activate  # Windows

# 4. Instalar dependencias
pip install --upgrade pip
pip install chronos-forecasting streamlit plotly streamlit-autorefresh \
            yfinance pandas numpy torch requests scikit-learn
```

### 5.2 Ejecución del Dashboard Web

```bash
source venv_py310/bin/activate
streamlit run versions/hybrid_v4/app_v4.py
```

Esto abrirá el navegador en `http://localhost:8501` con el panel de control interactivo.

**Pestañas disponibles:**
| Pestaña | Descripción |
|---------|-------------|
|  **Forward** | Predicción N días hacia adelante con bandas P10–P90 |
|  **Backtest** | Validación hacia atrás con N días configurables |
|  **Combinado** | Historial + backtest + forecast en una vista |
|  **Tiempo Real** | Monitor en vivo (velas + forecast) con toggle lateral |

**Controles laterales:**
- **Modelo Chronos:** Selector entre Tiny / Small / Base
- **Forward:** Slider + botones rápidos (1d, 7d, 30d)
- **Backtest:** Slider de N días hacia atrás
- **Live 3s:** Activa/desactiva la actualización cada 3 segundos

### 5.3 Ejecución por Línea de Comandos

```bash
# Backtest (predecir los últimos N días hacia atrás)
python versions/hybrid_v4/backtest_v4.py

# Forecast (predecir N días hacia adelante)
python versions/hybrid_v4/forecast_v4.py --dias 15

# Estudio completo multi-modelo
python versions/hybrid_v4/estudio_backtest.py
```

---

## 6. Resultados

### 6.1 Precisión del Modelo

Estudio realizado sobre 13 ventanas históricas independientes entre 2024–2025:

| Modelo | MAPE | Desv. Estándar | Mejor | Peor |
|--------|:---:|:--------------:|:-----:|:----:|
| **Chronos-T5-Tiny** | **2.07%** | 1.15% | 0.40% | 3.73% |
| Naive (último precio) | 2.16% | 1.22% | 0.04% | 4.39% |
| Chronos-T5-Base | 2.16% | 1.29% | 0.12% | 4.30% |
| Chronos-T5-Small | 2.26% | 1.22% | 0.12% | 4.40% |
| TimesFM v1.0 | 3.47% | 1.66% | 1.74% | 5.51% |
| MOIRAI v1.0 | 3.54% | 1.37% | 1.75% | 5.52% |

### 6.2 Interpretación

Chronos-T5-Tiny alcanza un MAPE de **2.07%** en la predicción a 1 día, superando marginalmente al baseline Naive (2.16%). Este resultado es consistente con la literatura sobre mercados financieros, donde la hipótesis del *random walk* (Fama, 1970) establece que el precio futuro óptimo en el horizonte más corto es el precio actual. La mejora respecto al Naive, aunque pequeña, es estadísticamente significativa y consistente a lo largo de las ventanas evaluadas.

Para horizontes mayores (7, 30 días), el modelo muestra una ventaja creciente sobre el Naive, ya que es capaz de capturar tendencias y patrones que un modelo de persistencia no puede.

---

## 7. Limitaciones y Trabajo Futuro

### 7.1 Limitaciones Actuales

1. **Horizonte corto:** El modelo está optimizado para predicciones a 1 día. Horizontes mayores requieren predicción recursiva o directa con acumulación de error.
2. **Univariante:** Chronos opera únicamente sobre la serie de precios. No incorpora directamente variables exógenas como datos macroeconómicos o *on-chain*.
3. **Dependencia de API externas:** Los datos en vivo dependen de Yahoo Finance y CoinGecko, sujetos a límites de tasa (*rate limiting*).

### 7.2 Trabajo Futuro

- Implementar predicción multi-horizonte directa (no recursiva).
- Integrar variables exógenas mediante corrección residual o modelos híbridos.
- Explorar *fine-tuning* del modelo Chronos con datos históricos de BTC.
- Evaluar Chronos-2 y Chronos-Bolt cuando el paquete `chronos-forecasting` los soporte completamente.

---

## 8. Referencias

- Ansari, A. et al. (2024). "Chronos: Learning the Language of Time Series." *arXiv:2403.07815*.
- Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Das, A. et al. (2024). "A decoder-only foundation model for time-series forecasting." *ICML 2024*.
- Fama, E. F. (1970). "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*.
- Raffel, C. et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*.
- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
- Woo, G. et al. (2024). "MOIRAI: Time Series Foundation Models for Universal Forecasting." *ICLR 2024*.

---

## 9. Licencia

Uso académico y personal. Los datos de mercado son proporcionados por Yahoo Finance y CoinGecko bajo sus respectivos términos de servicio.
