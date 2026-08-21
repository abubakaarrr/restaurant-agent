"""Application services shared by LangGraph and managed voice providers."""

from app.services.restaurant import RestaurantService, restaurant_service

__all__ = ["RestaurantService", "restaurant_service"]
