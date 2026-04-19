import sys
import traceback

def run_tests():
    try:
        import serving.api
        print("serving.api loaded")
        from ml_platform.registry import ModelRegistry
        print("ModelRegistry loaded")
        from monitoring.drift_simulator import DriftSimulator
        print("DriftSimulator loaded")
        from lifecycle.trainer import AutoRetrainer
        print("AutoRetrainer loaded")
        import scripts.generate_demo_data
        print("generate_demo_data loaded")
        print("ALL_IMPORTS_SUCCESS")
    except Exception as e:
        print("IMPORT_ERROR")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
