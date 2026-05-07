from .synthetic import generate_synthetic_hazard


def generate_hazard(config: dict, T: int):
    """Dispatcher matching HANDOFF §6: generate_hazard(config, T) -> dict."""
    source = config["hazards"]["source"]
    if source == "synthetic":
        return generate_synthetic_hazard(config, T)
    if source == "real":
        from .real import generate_real_hazard  # M3
        return generate_real_hazard(config, T)
    raise ValueError(f"Unknown hazards.source: {source!r}")


__all__ = ["generate_hazard", "generate_synthetic_hazard"]
