adding ticker to app. see below procedure:
Because the app’s watchlist box is loaded from this code:

DEFAULT_WATCHLIST = [
    "NVDA", "SMH", "VGT", "QQQM", "VOO",
    "AMD", "AVGO", "MSFT", "META", "GOOGL",
    "TSM", "MU", "ARM", "OXY", "MO", "HRL"
]

So after rerun/reload, it goes back to this default list. Since VUG is not inside DEFAULT_WATCHLIST, it disappears.

Fix

In GitHub, edit:

ai_stock_mobile_app.py

Find DEFAULT_WATCHLIST, and add "VUG":

DEFAULT_WATCHLIST = [
    "NVDA", "SMH", "VGT", "VUG", "QQQM", "VOO",
    "AMD", "AVGO", "MSFT", "META", "GOOGL",
    "TSM", "MU", "ARM", "OXY", "MO", "HRL"
]

Then:

Commit changes
Go to Streamlit Cloud
Reboot app or Deploy latest commit

After that, VUG will always appear in Watchlist Settings.
