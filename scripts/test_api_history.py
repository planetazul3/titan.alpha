
import asyncio
import asyncio
from data.ingestion.client import DerivClient
from config.settings import load_settings
import datetime

async def test_call(client, name, params):
    print(f"\n--- {name} ---")
    print(f"Params: {params}")
    try:
        res = await client.api.ticks_history(params)
        if 'error' in res:
            print(f"API Error: {res['error']}")
            return
        
        history = res.get('history', {})
        candles = res.get('candles', [])
        
        if history:
            times = history.get('times', [])
            if times:
                print(f"Ticks returned: {len(times)}")
                print(f"First: {times[0]} ({datetime.datetime.fromtimestamp(times[0])})")
                print(f"Last:  {times[-1]} ({datetime.datetime.fromtimestamp(times[-1])})")
            else:
                print("History key present but empty times")
        elif candles:
            print(f"Candles returned: {len(candles)}")
            print(f"First: {candles[0]['epoch']} ({datetime.datetime.fromtimestamp(candles[0]['epoch'])})")
            print(f"Last:  {candles[-1]['epoch']} ({datetime.datetime.fromtimestamp(candles[-1]['epoch'])})")
        else:
            print("No history or candles returned")
    except Exception as e:
        print(f"Execution Error: {e}")


async def main():
    settings = load_settings()
    # token variable was unused; DerivClient uses settings.deriv_api_token
    client = DerivClient(settings=settings)
    await client.connect()
    
    # 1. Test R_100 with only end and count (string)
    await test_call(client, "R_100: end only (string), count 10", {
        'ticks_history': 'R_100',
        'end': '1704067200', # 2024-01-01
        'style': 'ticks',
        'count': 10
    })

    # 2. Test R_100 with only end and count (int)
    await test_call(client, "R_100: end only (int), count 10", {
        'ticks_history': 'R_100',
        'end': 1704067200,
        'style': 'ticks',
        'count': 10
    })

    # 3. Test R_100 with start and end (no count)
    await test_call(client, "R_100: start + end, no count", {
        'ticks_history': 'R_100',
        'start': 1704067200, # 2024-01-01
        'end': 1704070800,   # 1 hour later
        'style': 'ticks'
    })

    # 4. Test R_100 with a VERY RECENT range (Dec 1, 2025)
    await test_call(client, "R_100: RECENT (Dec 1, 2025)", {
        'ticks_history': 'R_100',
        'start': 1733011200, # 2025-12-01
        'end': str(1733014800),   # 1 hour later
        'style': 'ticks'
    })

    # 5. Test R_100 with candles for Jan 2024
    await test_call(client, "R_100: candles (Jan 2024)", {
        'ticks_history': 'R_100',
        'start': 1704067200,
        'end': str(1704070800),
        'style': 'candles',
        'granularity': 60
    })

    # 6. Test R_100 with candles for Jan 2025 (Known success period)
    await test_call(client, "R_100: candles (Jan 2025)", {
        'ticks_history': 'R_100',
        'start': 1735689600, # 2025-01-01
        'end': str(1735693200),   # 1 hour later
        'style': 'candles',
        'granularity': 60
    })

if __name__ == "__main__":
    asyncio.run(main())
