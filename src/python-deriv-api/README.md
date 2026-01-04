# python-deriv-api

A Python implementation of the Deriv API library, updated for modern Python environments.

## Requirements

*   **Python**: 3.9 or higher is recommended.
*   **Dependencies**:
    *   `websockets>=14.0` (Updated for better performance and stability)
    *   `reactivex>=4.0`
    *   `typing_extensions` (for Python < 3.10)

## Installation

### From Source (Local Development)

If you have downloaded this source code, you can install it directly using pip:

```bash
cd python-deriv-api
pip install .
```

Or install it in editable mode if you plan to modify it:

```bash
pip install -e .
```

## Usage

This library simplifies WebSocket connections and API calls to Deriv.

### Basic Example

```python
import asyncio
from deriv_api import DerivAPI

async def sample():
    api = DerivAPI(app_id=1089) # Replace with your App ID
    
    # Ping
    response = await api.ping({'ping': 1})
    print(response)

    # Authorize (if needed)
    # await api.authorize('YOUR_TOKEN')

    # Get Balance
    # balance = await api.balance()
    # print(balance)

    await api.clear()

if __name__ == "__main__":
    asyncio.run(sample())
```

## Advanced Connection

You can manage the WebSocket connection manually if preferred:

```python
import asyncio
import websockets
from deriv_api import DerivAPI

async def manual_connect():
    connection = await websockets.connect('wss://ws.derivws.com/websockets/v3?app_id=1089')
    api = DerivAPI(connection=connection)
    
    # ... usage ...
    
    await api.clear()
```

## Development

### Running Tests

To run the test suite, ensure you have `pytest` installed:

```bash
pip install pytest pytest-asyncio pytest-mock
pytest
```

## Documentation

The official API reference can be found at [api.deriv.com](https://api.deriv.com/).
