import kaggle

datasets = [
    "jeffaudi/coco-2014-dataset-for-yolov3",
    "romainbeaumont/laion400m"
]

kaggle.api.authenticate()

for repo in datasets:
    kaggle.api.dataset_download_files(
        repo,
        path="../data/datasets/",
        unzip=True
    )
