# 🌅 Should I Go See the Sunset?

Ever stared at the horizon two hours before sunset, wondering if you’ll be blessed by a glorious sky… or just catch smoggy orange mush? Worry no more! This highly scientific, probably overengineered project evaluates sunset potential using images taken at **T−2h and T−1h**. It predicts—on a 1-5 scale—whether it’s worth dropping everything and sprinting to your nearest west-facing beach.

## What Is This?

A PyTorch-based dual-image classifier that consumes sunset images taken **2 hours** and **1 hour** before sunset, and predicts a sunset "glory score" (1-5).

Yes, it's trained on real data. Yes, there's a GUI. No, we don't guarantee enlightenment, but you might achieve it anyway.

## Main Scripts

| File                     | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `create_sunset_model.py` | Builds, trains, evaluates, and predicts sunset quality using a model.   |
| `order_images.py`        | Organizes raw images into a structured JSON file for training.          |
| `sunset_images_grouped.json` | Output file that links image pairs and their manually assigned scores.  |
| `img/ob_hotel/`          | Directory where sunset image files live (not included here).            |
| `model/`                 | Folder for saving/loading trained models.                              |

## Model Architecture

- **Dual ResNet-18 backbones** (shared or separate) 🤖
- Processes -2h and -1h images separately → feature concat → classification
- Outputs a score from 1 (💩) to 5 (🔥) based on sunset beauty

## GUI Mode (Because Buttons > CLI)

Modes:
- Train
    - Train the model with one click
- Predict
    - Choose two images: one from -2h, one from -1h
    - Press "Predict Score"

## 🧪 Training

Make sure `sunset_images_grouped.json` is populated with:
- At least one pair of images for each date (`type`: "-2h" and "-1h")
- A `score` between 1 and 5

Training parameters:
- `ResNet18` ×2
- 10 epochs
- 64×64 images
- Batch size: 32
- Optimizer: Adam
- Loss: CrossEntropy

## 🧙 Data Preparation

Run `order_images.py` to generate/update your training JSON file based on existing `.jpg` images named like:

```
25_07_24--18_45_00.jpg
```

## 🧙‍♂️ Why?

Fun excuse to play with:
- Paired-image classification
- PyTorch
- PyQt6
- Sunset FOMO

## TODO

- [ ] Automatically grab data via some google cloud or PA or something
- [ ] Host somewhere and have it send a message that the sunset will be fire

## License

MIT. Use it freely. Don’t sue if you miss the best sunset of your life.