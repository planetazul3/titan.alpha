import asyncio
from deriv_api import DerivAPI
from reactivex import operators as op

async def main():
    print("Connecting...")
    api = DerivAPI(app_id=1089)

    symbol = "R_100"
    print(f"Subscribing to ticks for {symbol}...")
    
    # Subscribe to ticks
    source_ticks = await api.subscribe({'ticks': symbol})
    
    # We want to just take 5 ticks and print them
    print("Waiting for 5 ticks...")
    
    # Use ReactiveX to handle the stream
    # take(5) -> completes after 5 items
    # to_list() -> collects them into a list
    # to_future() -> allows us to await the result
    ticks_list = await source_ticks.pipe(
        op.take(5),
        op.to_list(),
        op.to_future()
    )

    for i, tick_data in enumerate(ticks_list, 1):
        quote = tick_data['tick']['quote']
        epoch = tick_data['tick']['epoch']
        print(f"Tick {i}: {quote} at {epoch}")

    print("Done.")
    await api.clear()

if __name__ == "__main__":
    asyncio.run(main())
