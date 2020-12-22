from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
<<<<<<< HEAD
trainer.setDataDirectory(data_directory="birds-dataset")
=======
trainer.setDataDirectory(data_directory="sample-photos")
>>>>>>> 1e7c04db0412900a9ca3f6cd589eaa1f01e75239
trainer.setTrainConfig(object_names_array=["nest"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

# metrics = trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
<<<<<<< HEAD
# print(metrics)
=======
# print(metrics)
>>>>>>> 1e7c04db0412900a9ca3f6cd589eaa1f01e75239
