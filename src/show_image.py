import argparse

from PIL import Image
import matplotlib.pyplot as plt

def show_image(path, label, confidence):
    img = Image.open(path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{label} ({confidence:.2f})")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Image path")
    parser.add_argument("label", type=str, help="Label")
    parser.add_argument("confidence", type=float, help="Confidence")
    args = parser.parse_args()

    show_image('PetImages/' + args.path, args.label, args.confidence)
