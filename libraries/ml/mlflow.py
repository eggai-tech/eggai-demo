from pathlib import Path

import dotenv
from mlflow import MlflowClient

dotenv.load_dotenv()


def find_model(model_name: str, version: str, artifact_path: str = "model") -> str:
    cache_path = Path(__file__).resolve().parents[1] / ".cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    pickle_path = cache_path / model_name / version / artifact_path / "model.pkl"
    if pickle_path.exists():
        return str(pickle_path)
    pytorch_path = (
        cache_path / model_name / version / artifact_path / "data" / "model.pth"
    )
    if pytorch_path.exists():
        return str(pytorch_path)

    client = MlflowClient()
    mv = client.get_model_version(name=model_name, version=version)
    run_id = mv.run_id
    if run_id is None:
        raise ValueError(
            f"Model version {model_name} v{version} has no associated run to "
            "download artifacts from"
        )

    dest = cache_path / model_name / version
    dest.mkdir(parents=True, exist_ok=True)
    client.download_artifacts(run_id, artifact_path, dst_path=str(dest))
    if pickle_path.exists():
        return str(pickle_path)
    if pytorch_path.exists():
        return str(pytorch_path)

    raise FileNotFoundError(
        f"No model.pkl or data/model.pth under {artifact_path} for "
        f"{model_name} v{version} after downloading to {dest}"
    )


if __name__ == "__main__":
    print(find_model("attention_net_0.25_0.0002", version="1"))
