# Parámetros de Mercado para Predicción de Bitcoin
## Rankeados de Mayor a Menor Impacto Potencial

### Impacto MUY ALTO (MACRO + CORRELACIÓN DIRECTA)
| # | Parámetro | Fuente API | Costo | Cómo obtenerlo |
|---|---|---|---|---|
| 1 | **S&P 500 (^GSPC)** | Yahoo Finance (`yfinance`) | Gratis | `yf.download("^GSPC")` |
| 2 | **DXY — Índice Dólar (DX-Y.NYB)** | Yahoo Finance (`yfinance`) | Gratis | `yf.download("DX-Y.NYB")` |
| 3 | **Oro (GC=F)** | Yahoo Finance (`yfinance`) | Gratis | `yf.download("GC=F")` |
| 4 | **Tasa 10 años USA (^TNX)** | Yahoo Finance (`yfinance`) | Gratis | `yf.download("^TNX")` |
| 5 | **Volumen BTC en exchanges** | CoinGecko API | Gratis | `requests.get("https://api.coingecko.com/api/v3/coins/bitcoin/tickers")` |

### Impacto ALTO (ON-CHAIN + SENTIMIENTO)
| # | Parámetro | Fuente API | Costo | Cómo obtenerlo |
|---|---|---|---|---|
| 6 | **Hash Rate (Poder minero)** | Blockchain.com | Gratis | `requests.get("https://api.blockchain.info/charts/hash-rate")` |
| 7 | **Direcciones Activas** | Blockchain.com | Gratis | `requests.get("https://api.blockchain.info/charts/n-active-addresses")` |
| 8 | **Fear & Greed Index** | alternative.me | Gratis | `requests.get("https://api.alternative.me/fng/")` |
| 9 | **Reservas BTC en Exchanges** | Coin Metrics / Glassnode | Limitado/Paid | Glassnode Studio API |
| 10 | **M2 Money Supply (Liquidez Global)** | FRED API | Gratis | `fred.get_series("M2SL")` |

### Impacto MEDIO (MACRO + TÉCNICO)
| # | Parámetro | Fuente API | Costo | Cómo obtenerlo |
|---|---|---|---|---|
| 11 | **CPI / Inflación** | FRED API | Gratis | `fred.get_series("CPIAUCSL")` |
| 12 | **VIX (Índice de Miedo)** | Yahoo Finance | Gratis | `yf.download("^VIX")` |
| 13 | **Open Interest Futuros** | CoinGlass / Binance API | Gratis | Web scraping CoinGlass |
| 14 | **Funding Rate Perpetuo** | Binance/Bybit API | Gratis | WebSocket REST de Bybit |
| 15 | **Google Trends "Bitcoin"** | pytrends | Gratis | `pytrends.trending_searches()` |
| 16 | **Transacciones On-Chain** | Blockchain.com | Gratis | `api.blockchain.info/charts/n-transactions` |

### Impacto BAJO (ESPECULATIVO + SOCIAL)
| # | Parámetro | Fuente API | Costo | Cómo obtenerlo |
|---|---|---|---|---|
| 17 | **Reddit r/bitcoin menciones** | Pushshift API | Gratis | `api.pushshift.io/reddit/submission/search` |
| 18 | **Twitter/X volumen** | Twitter API v2 | Limitado | `tweepy` con dev account |
| 19 | **Próximo Halving** | Calculado con block height | Gratis | Fijo cada 210,000 bloques |
| 20 | **Stock-to-Flow Ratio** | Calculado | Gratis | `block_reward / circulating_supply` |
| 21 | **Miner Revenue** | Blockchain.com | Gratis | `api.blockchain.info/charts/miners-revenue` |
| 22 | **Bitcoin Dominance** | CoinGecko API | Gratis | `api.coingecko.com/api/v3/global` |
| 23 | **Tamaño Promedio Bloque** | Blockchain.com | Gratis | `api.blockchain.info/charts/avg-block-size` |
| 24 | **Número de Wallets** | Blockchain.com | Gratis | `api.blockchain.info/charts/n-unique-addresses` |
| 25 | **Velocidad del Dinero BTC** | Coin Metrics | Limitado | Coin Metrics API |


### Impacto VARIABLE (EVENTOS DISCRETOS — Hackeos/Seguridad)
| # | Parámetro | Fuente API | Costo | Cómo obtenerlo |
|---|---|---|---|---|
| 26 | **Días desde último hack > $100M** | DeFiLlama / rekt.news | Gratis | Web scraping `api.llama.fi/hacks` → filtrar por BTC/ETH |
| 27 | **Monto total robado últimos 30 días** | DeFiLlama API | Gratis | Suma de montos de hacks recientes |
| 28 | **Flag de hack activo (0/1)** | Calculado | Gratis | 1 si hay un exploit activo en las últimas 48h |
| 29 | **Número de exchanges comprometidos** | DeFiLlama API | Gratis | Conteo de plataformas afectadas en el mes |
