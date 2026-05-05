# Freqtrade Binance Futures network troubleshooting

If the error contains:

```text
binance GET https://fapi.binance.com/fapi/v1/exchangeInfo
Markets were not loaded.
```

the config is already pointing to Binance USDS-M Futures. At this stage the usual cause is local network, proxy, firewall, or regional access restrictions, not the FreqAI strategy code.

## 1. Clear broken proxy variables in the same PowerShell window

Run this before `freqtrade download-data`:

```powershell
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY -ErrorAction SilentlyContinue
curl.exe -I https://fapi.binance.com/fapi/v1/exchangeInfo
```

In this workspace the proxy variables were detected as:

```text
HTTP_PROXY=http://127.0.0.1:9
HTTPS_PROXY=http://127.0.0.1:9
ALL_PROXY=http://127.0.0.1:9
```

That proxy endpoint is invalid unless you intentionally run a proxy server on port 9.

## 2. If you have a real local proxy

Use your actual proxy port. For example, if your proxy is `127.0.0.1:7890`, add these fields under both `exchange.ccxt_config` and `exchange.ccxt_async_config`:

```json
"httpsProxy": "http://127.0.0.1:7890",
"wsProxy": "http://127.0.0.1:7890"
```

Do not leave the proxy as `127.0.0.1:9`.

## 3. If Binance returns a regional restriction

If `curl.exe -I https://fapi.binance.com/fapi/v1/exchangeInfo` returns a regional or eligibility restriction, do not bypass it for live trading. Use an exchange and futures product that is legal and available in your location, then adjust the Freqtrade `exchange.name`, pair format, and futures settings accordingly.

## 4. Retry after fixing network/proxy

```powershell
freqtrade download-data `
  --userdir freqtrade_user_data `
  --config freqtrade_user_data/config_freqai.example.json `
  --timeframes 15m 1h 4h `
  --trading-mode futures `
  --timerange 20240101-
```

## 5. Windows aiohttp / aiodns DNS error

If `curl.exe` can return `200`, but Freqtrade still fails with:

```text
aiodns.error.DNSError: (11, 'Could not contact DNS servers')
aiohttp.client_exceptions.ClientConnectorDNSError
```

then Binance is reachable, but Python's async DNS resolver cannot contact your DNS server. On Windows, the simplest workaround is usually to remove `aiodns` from the exact Python environment used by Freqtrade so aiohttp falls back to the system resolver.

Use the Python path shown in the Freqtrade traceback:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m pip uninstall -y aiodns
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -c "import aiohttp.resolver as r; print(r.DefaultResolver)"
```

The second command should print `ThreadedResolver`. Then rerun `freqtrade download-data`.

If it still prints `AsyncResolver`, remove `pycares` too:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m pip uninstall -y pycares
```
