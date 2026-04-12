to train, run:
python train.py --task classification --epochs 20 --batch-size 64
python train.py --task localization --epochs 30 --pretrained-classifier ./checkpoints/classification.pth
python train.py --task segmentation --epochs 20 --pretrained-classifier ./checkpoints/classification.pth

to validate scores run:
python inference.py --task classification --checkpoint ./checkpoints/classification.pth
python inference.py --task localization --checkpoint ./checkpoints/localization.pth
python inference.py --task segmentation --checkpoint ./checkpoints/segmentation.pth

to run on real images, you can run:
python wandb_scripts.py --image-dir google-images --project DA6401_Assignment_2 --run-name google_images_predictions