import os

base_path = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages", "agents", "scripts")
if not os.path.exists(base_path):
    base_path = "/app/.venv/lib/python3.8/site-packages/agents/scripts"

target_file = os.path.join(base_path, "networks.py")

if not os.path.exists(target_file):
    print("❌ Could not find networks.py inside agents package.")
else:
    with open(target_file, "r", encoding="utf-8") as f:
        code = f.read()

    if "tf.contrib.distributions" in code:
        new_code = code.replace(
            "tf.contrib.distributions",
            "import tensorflow_probability as tfp\ntfd = tfp.distributions  # patched\n# tf.contrib.distributions"
        )
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(new_code)
        print("✅ Patched TensorFlow contrib import successfully!")
    else:
        print("⚠️ networks.py already patched or doesn't use tf.contrib.distributions")
