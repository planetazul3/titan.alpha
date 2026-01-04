import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

class SafetyReconciler:
    """Helper to reconcile safety state with API."""
    
    @staticmethod
    async def reconcile(client: Any, pending_store: Optional[Any] = None) -> dict[str, Any]:
        """
        Reconcile safety state with API on startup.
        
        Queries open contracts from the Deriv API and compares with the
        pending trade store to detect discrepancies.
        """
        result: dict[str, Any] = {
            "api_open_count": 0,
            "store_pending_count": 0,
            "matched": [],
            "api_only": [],
            "store_only": [],
            "error": None
        }
        
        try:
            # Get open contracts from API
            api_response = await client.get_open_contracts()
            
            # Parse contract IDs from API response
            api_contracts = set()
            if isinstance(api_response, dict):
                # proposal_open_contract returns a dict with contract details
                poc = api_response.get("proposal_open_contract", {})
                if poc and isinstance(poc, dict):
                    cid = poc.get("contract_id")
                    if cid:
                        api_contracts.add(str(cid))
            elif isinstance(api_response, list):
                for item in api_response:
                    if isinstance(item, dict):
                        cid = item.get("contract_id") or item.get("id")
                        if cid:
                            api_contracts.add(str(cid))
                            
            result["api_open_count"] = len(api_contracts)
            
            # Get pending trades from store
            store_contracts = set()
            if pending_store is not None:
                try:
                    pending = pending_store.get_all_pending()
                    for trade in pending:
                        cid = trade.get("contract_id")
                        if cid:
                            store_contracts.add(str(cid))
                except Exception as e:
                    logger.warning(f"Could not read pending store: {e}")
                    
            result["store_pending_count"] = len(store_contracts)
            
            # Compare
            result["matched"] = list(api_contracts & store_contracts)
            result["api_only"] = list(api_contracts - store_contracts)
            result["store_only"] = list(store_contracts - api_contracts)
            
            # Log reconciliation results
            if result["api_only"]:
                logger.warning(
                    f"⚠️ Reconciliation found {len(result['api_only'])} open contracts "
                    f"NOT in pending store: {result['api_only']}"
                )
            if result["store_only"]:
                logger.info(
                    f"Reconciliation: {len(result['store_only'])} stored trades "
                    f"already settled: {result['store_only']}"
                )
            
            logger.info(
                f"Reconciliation complete: {result['api_open_count']} API contracts, "
                f"{result['store_pending_count']} pending in store, "
                f"{len(result['matched'])} matched"
            )
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            result["error"] = str(e)
            
        return result
