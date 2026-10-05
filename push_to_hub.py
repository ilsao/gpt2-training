"""Upload the newest complete checkpoint of one finished run."""

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

from checkpoints import COMPLETE_MARKER, atomic_write_json, latest_complete_checkpoint

# Hub 1.x uses httpx; older installations use requests.
NETWORK_ERRORS = (ConnectionError, TimeoutError)
try:
    from httpx import TransportError
    NETWORK_ERRORS += (TransportError,)
except ImportError:
    pass
try:
    from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout
    NETWORK_ERRORS += (RequestsConnectionError, Timeout)
except ImportError:
    pass


def retryable(error):
    response = getattr(error, "response", None)
    if response is not None:
        return response.status_code in (408, 429) or 500 <= response.status_code < 600
    return isinstance(error, NETWORK_ERRORS)


def upload_checkpoint(run_dir, repo_id, api=None):
    run_dir = Path(run_dir)
    checkpoint_dir, manifest = latest_complete_checkpoint(run_dir)
    print(f"Uploading {checkpoint_dir} to {repo_id} (step {manifest['global_step']})", flush=True)
    api = api if api is not None else HfApi()
    for attempt in range(3):
        try:
            commit = api.upload_folder(
                repo_id=repo_id, repo_type="model", folder_path=str(checkpoint_dir),
                path_in_repo="", run_as_future=False,
                allow_patterns=[*manifest["files"], COMPLETE_MARKER],
                commit_message=f"Upload {run_dir.name} checkpoint-{manifest['global_step']}",
            )
            break
        except Exception as error:
            if attempt == 2 or not retryable(error):
                raise
            delay = (10, 30)[attempt]
            print(f"Temporary upload failure; retrying in {delay}s ({attempt + 1}/3)", flush=True)
            time.sleep(delay)
    atomic_write_json(run_dir / "upload-result.json", {
        "repo_id": repo_id, "global_step": manifest["global_step"],
        "checkpoint": checkpoint_dir.name, "commit_oid": commit.oid,
        "commit_url": commit.commit_url,
    })
    print(f"Uploaded checkpoint: {commit.commit_url}", flush=True)
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", default=os.environ.get("HF_HUB_REPO", "asciibase64/gpt2-c4"))
    args = parser.parse_args()
    upload_checkpoint(args.run_dir, args.repo_id)


if __name__ == "__main__":
    main()
