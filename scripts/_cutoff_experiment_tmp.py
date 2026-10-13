import subprocess, sys, time
from src.data.features.minutes_feature_manager import MinutesFeatureManager
from src.data.features.combined_feature_manager import CombinedFeatureManager

if __name__ == "__main__":
    for tag in ("b1542", "b1558"):
        t = time.time()
        m = MinutesFeatureManager(f"config/feature/minutes_feature_config_{tag}.yaml").calculate_features()
        print(f"MINUTES {tag}: {len(m)} rows, {m['date'].nunique()} days ({time.time()-t:.0f}s)", flush=True)
        c = CombinedFeatureManager(f"config/feature/combined_feature_config_{tag}.yaml").combine()
        print(f"COMBINED {tag}: {len(c)} rows, {c['date'].nunique()} days", flush=True)
        with open(f"logs/{tag}_walk_forward.log", "w") as log:
            code = subprocess.call([sys.executable, "-m", "src.walk_forward", "--config", f"config/train/train_model_{tag}.yaml",
                                    "--variant", "weekly_plain_label", "--variant", "weekly_plain_label_from2025-06"], stdout=log, stderr=subprocess.STDOUT)
        print(f"WALK_FORWARD {tag} exit {code} ({time.time()-t:.0f}s total)", flush=True)
    print("EXPERIMENT DONE", flush=True)
