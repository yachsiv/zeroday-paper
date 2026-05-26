"""Data clients (Polygon, FlashAlpha, CBOE).

All clients are leak-free: historical methods do not return data published after
the requested timestamp. This is essential for honest replay.
"""
