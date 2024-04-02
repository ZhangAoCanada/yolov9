import os
import clip 
import torch
import PIL
from tqdm import tqdm
import shutil

class ClipLoss:
    def __init__(self, class_names):
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.model.cuda().eval()
        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.visual.output_dim
        self.class_names = class_names
        self.text_features = self.to_text_features()
    
    def to_text_features(self, ):
        text_descriptions = [f"This is a photo of a {class_name}" for class_name in self.class_names]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def __call__(self, pred_box, pred_scores, img):
        """
        pred_box: (bs, num_total_anchors, 4)
        pred_scores: (bs, num_total_anchors, num_classes)
        img: (bs, 3, h, w)
        """
        save_dir = "/home/za/Documents/yolov9/debugging"
        # remove all files in save dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        all_features = []
        for i in tqdm(range(pred_box.size(0))):
            # transfer image to PIL.Image
            # image = img[i].cpu().numpy().transpose(1, 2, 0)
            # image = PIL.Image.fromarray((image * 255).astype("uint8"))
            for j in tqdm(range(pred_box.size(1))):
                if pred_scores[i, j].max() < 0.5:
                    continue
                x1, y1, x2, y2 = pred_box[i, j]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, img.size(3)), min(y2, img.size(2))
                img_box = img[i, :, y1:y2, x1:x2]
                image = img_box.cpu().numpy().transpose(1, 2, 0)
                image = PIL.Image.fromarray((image * 255).astype("uint8"))
                image.save(os.path.join(save_dir, f"image_{i}_{j}.jpg")) # save for debugging
                image_input = self.preprocess(image).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_probs = (100.0 / 100) * image_features @ self.text_features.T.softmax(dim=-1)
                    print("[DEBUG] text_probs.shape: ", text_probs.size())
                    all_features.append(image_features)
