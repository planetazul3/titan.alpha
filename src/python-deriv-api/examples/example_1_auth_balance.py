import asyncio
import os
import sys
from deriv_api import DerivAPI

async def main():
    # 1. Check for token
    token = os.environ.get('DERIV_TOKEN')
    if not token:
        print("Please set DERIV_TOKEN environment variable")
        return

    # 2. Connect
    print("Connecting...")
    api = DerivAPI(app_id=1089)

    # 3. Authorize
    print("Authorizing...")
    auth = await api.authorize(token)
    print(f"Hello, {auth['authorize']['email']}!")

    # 4. Get Balance
    balance = await api.balance()
    print(f"Current Balance: {balance['balance']['balance']} {balance['balance']['currency']}")

    # 5. Cleanup
    await api.clear()

if __name__ == "__main__":
    asyncio.run(main())
