from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="sample-photos")
trainer.setTrainConfig(object_names_array=["nest"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

# metrics = trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
# print(metrics)