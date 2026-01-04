import asyncio
import os
from deriv_api import DerivAPI

async def main():
    token = os.environ.get('DERIV_TOKEN')
    if not token:
        print("Please set DERIV_TOKEN environment variable")
        return

    print("Connecting...")
    api = DerivAPI(app_id=1089)

    print("Authorizing...")
    await api.authorize(token)

    # 1. Get Proposal
    # A simple Rise/Fall (CALL) contract on Volatility 100 Index
    print("Requesting proposal...")
    proposal_req = {
        "proposal": 1,
        "amount": 10,  # Stake
        "basis": "stake",
        "contract_type": "CALL",
        "currency": "USD",
        "duration": 5,
        "duration_unit": "t", # 5 ticks
        "symbol": "R_100"
    }
    
    proposal_resp = await api.proposal(proposal_req)
    prop_id = proposal_resp['proposal']['id']
    price = proposal_resp['proposal']['ask_price']
    print(f"Proposal ID: {prop_id}, Price: {price}")

    # 2. Buy Contract
    print("Buying contract...")
    # buy parameter uses the proposal ID
    buy_resp = await api.buy({"buy": prop_id, "price": price})
    
    contract_id = buy_resp['buy']['contract_id']
    print(f"Contract Bought! ID: {contract_id}")
    print(f"Longcode: {buy_resp['buy']['longcode']}")

    # 3. Wait for outcome (optional, just waiting a bit here)
    print("Waiting for contract to finish (approx 5-10s)...")
    await asyncio.sleep(8)
    
    # Check profit/loss (getting contract details)
    # We use proposal_open_contract to track it, but for simplicity here we just exit
    
    print("Done.")
    await api.clear()

if __name__ == "__main__":
    asyncio.run(main())
