"""Backward-compat re-exports.

SportsFeed base class now lives in sports.base.
BallDontLieFeed now lives in sports.nba.feed.
"""
from sports.base import SportsFeed
from sports.nba.feed import BallDontLieFeed

__all__ = ["BallDontLieFeed", "SportsFeed"]
