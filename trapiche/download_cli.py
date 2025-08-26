from . import model_registry

def main():
    print("Downloading/validating Trapiche models ...")
    try:
        model_registry.download_models()
        print(f"Models ready under {model_registry.cache_root()}")
    except Exception as e:
        print(f"Failed: {e}")
        raise SystemExit(1)
