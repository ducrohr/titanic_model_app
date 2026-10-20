# Titanic survival model

This repository contains a scikit-learn logistic regression model for predicting Titanic survival and a small Flask demo. Deeploy deploys the **model artifact**, not the Flask web page.

## Deeploy deployment

The repository follows Deeploy's [repository contract](https://docs.deeploy.ai/docs/repository/repository-requirements): `model/model.joblib` is the single model artifact in the `model` folder. It is trained with scikit-learn 1.5.1 and joblib 1.4.0, the versions listed for the current scikit-learn serving image in Deeploy's [supported versions](https://docs.deeploy.ai/docs/supported-versions). The optional `explainer` and `transformer` folders are not needed.

1. Commit and push this repository, including `model/model.joblib`, to GitHub.
2. In Deeploy, link the Git repository to a workspace. For a public repository, use its HTTPS URL; for a private repository, use its SSH URL and add Deeploy's generated read-only deploy key.
3. Create a managed deployment from this repository and select the **scikit-learn** framework. Select a serving image compatible with scikit-learn 1.5.1 if the Deeploy instance offers a choice.
4. Send inference requests using Deeploy's [V1 data format](https://docs.deeploy.ai/docs/api/inference-data-payload). For example:

   ```json
   {"instances": [[3, 1, 22]]}
   ```

   The three numbers are `pclass` (1–3), `sex` (male = 0, female = 1), and `age` in years, in that order. The prediction is `0` for did not survive or `1` for survived. A typical response has the form `{"predictions": [0]}` or `{"predictions": [1]}`.

The deployed endpoint address and authentication depend on the Deeploy workspace. The local `/predict` HTML form is a separate demo and is not the managed Deeploy inference endpoint.

## Local setup

Use Python 3.10–3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000` to use the Flask demo. The checked-in model is already trained.

To retrain it, run:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Training uses the checked-in `data/titanic.csv` from the [Seaborn sample datasets](https://github.com/mwaskom/seaborn-data), so it needs no network access. After retraining, commit the updated `model/model.joblib` before creating a new Deeploy deployment.
