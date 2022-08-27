# Hook Antrophometrics Module

Python API

## Resources

| Dependency Name | Documentation                | Description                                                                            |
| --------------- | ---------------------------- | -------------------------------------------------------------------------------------- |
| FastAPI         | https://fastapi.tiangolo.com | FastAPI framework, high performance, easy to learn, fast to code, ready for production |
---

## Run Locally

Follow [Pytorch's documentation](https://pytorch.org/get-started/locally/) to install Pytorch

To run locally in debug mode run:

```
uvicorn app.api:app --reload
```
Open your browser to http://localhost:8000/docs to view the OpenAPI UI.


For an alternate view of the docs navigate to http://localhost:8000/redoc

---

## Deploy with Azure Pipelines
Follow this guide to setup an Azure Resource Group with instances of Azure Kubernetes Service and Azure Container Registry and setup CI / CD with Azure Pipelines.

https://docs.microsoft.com/en-us/azure/devops/pipelines/ecosystems/kubernetes/aks-template?view=azure-devops
