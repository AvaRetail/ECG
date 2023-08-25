from datasets import load_dataset
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import os
from PIL import Image
import torch
import glob

def process_img_with_model(path, processor, model):
    pilImage = Image.open(path)
    # torchArr = 
    _normalized = Normalize(processor.image_mean, processor.image_std)
    h, w = processor.size.height, processor.size.width
    _compose = Compose([Resize((h, w)),
                        ToTensor(),
                        _normalized])
    
    processed_img =  _compose(pilImage.convert("RGB"))
    res = model(processed_img)
    print(res)
    return

def main():
    # model = AutoModelForImageClassification.from_pretrained(r"..\weights\viT-b-16\checkpoint-3957\pytorch_model.bin")
    # processor = AutoImageProcessor.from_pretrained(r"..\weights\vit-b-16\checkpoint-3957\preprocessor_config.json")

    # if os.path.isfile(args.ds) :
    #     process_img_with_model(args.ds, processor, model)

    # elif os.path.isdir(args.ds):
    #     for path in glob.glob(args.ds):
    #           process_img_with_model(args.ds, processor, model)
    classifier = pipeline("image-classification", model = r"C:\Users\ATI-G2\Documents\python\ECG\Transformers\checkpoint")
    img = Image.open(args.ds)
    res = classifier(img)
    print(res)
    # data = load_dataset(r"C:\Users\ATI-G2\Documents\python\ECG\data\vit-data\train",split="train[:10]") 


def main2():
    model = AutoModelForImageClassification.from_pretrained(r"C:\Users\ATI-G2\Documents\python\ECG\Transformers\checkpoint")
    img_processor = AutoImageProcessor.from_pretrained(r"C:\Users\ATI-G2\Documents\python\ECG\Transformers\checkpoint")

    img = Image.open(args.ds)
    inputs = img_processor(img, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

        print(logits)


class getArgs:
    ds = r"C:\Users\ATI-G2\Documents\python\ECG\data\rough\1dAVb\3701.jpg"

if __name__=="__main__":
    args = getArgs()
    # main()
    main2()