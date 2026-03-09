Terminal 1 — Start the policy server

uv run scripts/serve_policy.py --env LIBERO
This loads the pi05_libero config and checkpoint by default. To use a custom checkpoint:


uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /path/to/your/checkpoint
Terminal 2 — Run the LIBERO client
The LIBERO simulation needs a separate Python 3.8 environment:


# Create the venv (one-time)
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
Then run on libero_90:


python examples/libero/main.py --task-suite-name libero_90