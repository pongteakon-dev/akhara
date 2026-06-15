import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
import csv
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet import UNetModel
import wandb
from torchvision import transforms
from feature_extractor import ImageEncoder
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer


def resolve_device(requested_device: str) -> torch.device:
    requested_device = (requested_device or "auto").strip().lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested_device)

    if requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(requested_device)


torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95 #+ 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 960     # 960: Akhara, 256: IAM
IMG_HEIGHT = 384    # 384: Akhara, 64: IAM

c_classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}

### Borrowed from GANwriting ###
# def label_padding(labels, letter2index, num_tokens, pad_token, output_max_len):
#     ll = [letter2index[i] for i in labels]
#     ll = np.array(ll) + num_tokens
#     ll = list(ll)
#     num = output_max_len - len(ll)
#     if num > 0:
#         ll.extend([pad_token] * num)
#     return ll
#
#
# def build_label_dictionary(character_classes, write_json=True):
#     letter2index = {label: n for n, label in enumerate(character_classes)}
#     index2letter = {v: k for k, v in letter2index.items()}
#     if write_json:
#         with open("letter2index.json", "w", encoding="utf-8") as f:
#             json.dump(letter2index, f, ensure_ascii=False)
#         with open("index2letter.json", "w", encoding="utf-8") as f:
#             json.dump(index2letter, f, ensure_ascii=False)
#     return len(character_classes), letter2index, index2letter
#
#
# char_classes, letter2index, index2letter = build_label_dictionary(list(c_classes))
# tok = False
# if not tok:
#     tokens = {"PAD_TOKEN": char_classes}
# else:
#     tokens = {"GO_TOKEN": char_classes, "END_TOKEN": char_classes + 1, "PAD_TOKEN": char_classes + 2}
# num_tokens = len(tokens.keys())
# print('num_tokens', num_tokens)
#
#
# print('num of character classes', char_classes)
# vocab_size = char_classes + num_tokens

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    # create json object from dictionary if you want to save writer ids
    json_dict_l = json.dumps(letter2index)
    l = open("letter2index.json","w")
    l.write(json_dict_l)
    l.close()
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    l = open("index2letter.json","w")
    l.write(json_dict_i)
    l.close()
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)


print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens



def setup_logging(args):
    #os.makedirs("models", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def save_images(images, path, args, **kwargs):
    #print('image', images.shape)
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im


def save_single_images(images, path, args):
    if images.dim() == 4:
        image = images[0]
    else:
        image = images

    if args.latent is True:
        im = torchvision.transforms.ToPILImage()(image)
        if args.color is False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = image.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

def crop_whitespace_width(img):
    #tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    #rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


class KhmerDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, basefolder, fixed_size=(IMG_HEIGHT, IMG_WIDTH), transforms=None, seed=42):
        self.csv_path = csv_path
        self.basefolder = basefolder
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.seed = seed

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        self.writer_ids = sorted({int(r["writer_id"]) for r in self.rows})
        self.writer_id_to_index = {wid: idx for idx, wid in enumerate(self.writer_ids)}
        self.wclasses = len(self.writer_ids)

    def __len__(self):
        return len(self.rows)

    def _load_and_fit(self, rel_path):
        full_path = os.path.join(self.basefolder, rel_path)
        img = Image.open(full_path).convert("RGB")
        if img.width < IMG_WIDTH:
            img = ImageOps.pad(img, size=(IMG_WIDTH, IMG_HEIGHT), color="white")
        img = image_resize_PIL(img, height=int(1.0 * IMG_HEIGHT), width=int(1.0 * IMG_WIDTH))
        img = centered_PIL(img, (IMG_HEIGHT, IMG_WIDTH), border_value=255.0)
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __getitem__(self, index):
        row = self.rows[index]
        img_path = row["image_path"]
        transcr = row["label"]
        wid = int(row["writer_id"])
        wid_idx = self.writer_id_to_index[wid]

        img = self._load_and_fit(img_path)

        same_writer = [r for r in self.rows if int(r["writer_id"]) == wid and len(r["label"]) > 0]
        style_samples = random.sample(same_writer, k=min(5, len(same_writer)))
        style_imgs = [self._load_and_fit(s["image_path"]) for s in style_samples]
        style_imgs = torch.stack(style_imgs)

        cor_im = img

        return img, transcr, torch.tensor(wid_idx, dtype=torch.long), style_imgs, img_path, cor_im


def split_indices_by_writer(rows, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    writer_to_indices = {}
    for idx, row in enumerate(rows):
        wid = int(row["writer_id"])
        writer_to_indices.setdefault(wid, []).append(idx)

    train_indices = []
    val_indices = []
    for indices in writer_to_indices.values():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    return train_indices, val_indices


def build_style_images_for_writer(full_dataset, writer_id, count=5):
    matching_rows = [row for row in full_dataset.rows if int(row["writer_id"]) == int(writer_id)]
    if not matching_rows:
        raise ValueError(f"No samples found for writer_id={writer_id}")

    selected_rows = random.sample(matching_rows, k=min(count, len(matching_rows)))
    style_images = [full_dataset._load_and_fit(row["image_path"]) for row in selected_rows]
    return torch.stack(style_images)


def build_character_classes_from_csv(csv_path):
    res = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res.update(list(row["label"]))
    return sorted(list(res))


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    
class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(IMG_HEIGHT, IMG_WIDTH), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
        model.eval()
        tensor_list = []
        vae_model = vae.module if hasattr(vae, "module") else vae
        
        with torch.no_grad():
            pbar = tqdm(test_loader)
            style_feat = []
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]
                s_id = data[2].to(args.device)
                style_images = data[3].to(args.device)
                cor_im = data[5].to(args.device)
                img_path = data[4]
                
                
                if args.model_name == 'wordstylist':
                    #print('transcr', transcr)
                    batch_word_embeddings = []
                    for trans in transcr:
                        # word_embedding = label_padding(trans, letter2index, num_tokens, tokens["PAD_TOKEN"], OUTPUT_MAX_LEN)
                        word_embedding = label_padding(trans, num_tokens)
                        #print('word_embedding', word_embedding)
                        word_embedding = np.array(word_embedding, dtype="int64")
                        word_embedding = torch.from_numpy(word_embedding).long() 
                        batch_word_embeddings.append(word_embedding)
                    text_features = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    # text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=args.text_max_length).to(args.device)
                    text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                
                reshaped_images = style_images.reshape(-1, 3, IMG_HEIGHT, IMG_WIDTH)
                
                if style_extractor is not None:
                    style_features = style_extractor(reshaped_images).to(args.device)
                else:
                    style_features = None
            
                if args.latent == True:
                    x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                    
                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
                
                #scheduler
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:
                    
                    t_item = time.item()
                    t = (torch.ones(images.size(0)) * t_item).long().to(args.device)

                    with torch.no_grad():
                        noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                        prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                        x = prev_noisy_sample
                    
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae_model.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None, style_images_override=None):
        model.eval()
        tensor_list = []
        vae_model = vae.module if hasattr(vae, "module") else vae
        
        with torch.no_grad():
            style_images = None
            cor_im = False
            style_features = None
            text_features = x_text #[x_text]*n
            #print('text features', text_features.shape)
            # text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=args.text_max_length).to(args.device)
            text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            if style_images_override is not None:
                style_images = style_images_override.to(args.device)
                if style_extractor is not None:
                    reshaped_images = style_images.reshape(-1, 3, IMG_HEIGHT, IMG_WIDTH)
                    style_features = style_extractor(reshaped_images).to(args.device)
                else:
                    style_features = None
            elif args.img_feat == True:
                #pick random image according to specific style
                with open('./writers_dict_train.json', 'r') as f:
                    
                    wr_dict = json.load(f)
                reverse_wr_dict = {v: k for k, v in wr_dict.items()}
                with open('./utils/splits_words/iam_train_val.txt', 'r') as f:
                    train_data = f.readlines()
                    train_data = [i.strip().split(',') for i in train_data]
                    style_featur = []
                    for label in labels:
                        #print('label', label)
                        label_index = label.item()
    
                        matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index] and len(line[2])>3]

                        #pick the first 5 from matching lines
                        
                        if len(matching_lines) >= 5:
                            #five_styles = matching_lines[:5]
                            #pick first line and repeat
                            #five_styles = [matching_lines[0]]*5
                            five_styles = random.sample(matching_lines, 5)
                            #five_styles = matching_lines_style[:5]
                        else:
                            matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index]]
                            #print('matching lines', matching_lines)
                            # five_styles = matching_lines_style[:5]
                            five_styles = [matching_lines[0]]*5
                            #five_styles = random.sample(matching_lines, 5)
                        print('five_styles', five_styles)
                        #five_styles = random.sample(matching_lines, 5)
                        
                        cor_image_random = random.sample(matching_lines, 1)
                        #print('cor_image_random', cor_image_random)
                        #five_styles =[['a05/a05-084/a05-084-04-05.png', '000', 'which'], ['a03/a03-073/a03-073-04-04.png', '000', 'stage'], ['a01/a01-077u/a01-077u-02-02.png', '000', 'cables'], ['a05/a05-089/a05-089-00-05.png', '000', 'debate'], ['a05/a05-048/a05-048-00-00.png', '000', 'Long']] #class id 12
                        #five_styles = [['b06/b06-071/b06-071-08-06.png', '128', 'Labour'], ['b06/b06-019/b06-019-05-04.png', '128', 'West'], ['b06/b06-071/b06-071-05-03.png', '128', 'could'], ['c06/c06-027/c06-027-01-01.png', '128', 'advantage'], ['c06/c06-076/c06-076-01-05.png', '128', 'never']] #class id 1
                        
                        interpol = False
                        if interpol == True:
                            label2 = random.randint(0, 339) #random label
                            matching_lines2 = [line for line in train_data if line[1] == reverse_wr_dict[label2] and len(line[2])>3]
                            five_styles = random.sample(matching_lines2, 5)
                        #print('five_styles', five_styles)
                        #cor_image
                        fheight, fwidth = 64, 256
                        root_path = './iam_data/words'
                        cor_im = False
                        if cor_im == True:
                            cor_image = Image.open(os.path.join(root_path, cor_image_random[0][0])).convert('RGB') #['a05/a05-089/a05-089-00-05.png', '000', 'debate']
                            (cor_image_width, cor_image_height) = cor_image.size
                            cor_image = cor_image.resize((int(cor_image_width * 64 / cor_image_height), 64))
                            (cor_image_width, cor_image_height) = cor_image.size
                            
                            if cor_image_width < 256:
                                outImg = ImageOps.pad(cor_image, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                                cor_image = outImg
                            
                            else:
                                #reduce image until width is smaller than 256
                                while cor_image_width > 256:
                                    cor_image = image_resize_PIL(cor_image, width=cor_image_width-20)
                                    (cor_image_width, cor_image_height) = cor_image.size
                                cor_image = centered_PIL(cor_image, (64, 256), border_value=255.0)
                                    
                            cor_im_tens = transform(cor_image).to(args.device)
                            #print('cor image', cor_im_tens.shape)
                            cor_im_tens = cor_im_tens.unsqueeze(0)
                            cor_images = vae_model.encode(cor_im_tens.to(torch.float32)).latent_dist.sample()
                            cor_images = cor_images * 0.18215
                            
                        st_imgs = []
                        grid_imgs = []
                        for im_idx, random_f in enumerate(five_styles):
                            file_path = os.path.join(root_path, random_f[0])
                            #print('file_path', file_path)
                            
                            try:
                                img_s = Image.open(file_path).convert('RGB')
                            except ValueError:
                                # Handle the exception (e.g., print an error message)
                                print(f"Error loading image from {file_path}")
                                
                                # Find a replacement image that is not corrupted
                                replacement_idx = (im_idx + 1) % 5
                                replacement_f = five_styles[replacement_idx]
                                name = replacement_f[0] #.split(',')[1]
                                replacement_file_path = os.path.join(root_path, name)
                                img_s = Image.open(replacement_file_path).convert('RGB')
                                
                            (img_width, img_height) = img_s.size
                            img_s = img_s.resize((int(img_width * 64 / img_height), 64))
                            (img_width, img_height) = img_s.size
                            
                            if img_width < 256:
                                outImg = ImageOps.pad(img_s, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                                img_s = outImg
                            
                            else:
                                #reduce image until width is smaller than 256
                                while img_width > 256:
                                    img_s = image_resize_PIL(img_s, width=img_width-20)
                                    (img_width, img_height) = img_s.size
                                img_s = centered_PIL(img_s, (64, 256), border_value=255.0)
                            #make grid of all 5 images
                            #img_s = img_s.convert('L')
                            transform_tensor = transforms.ToTensor()
                            grid_im = transform_tensor(img_s)
                            grid_imgs += [grid_im]
                            
                            img_tens = transform(img_s).to(args.device)#.unsqueeze(0)
                            st_imgs += [img_tens]
                            #style_features = style_extractor(style_images).to(args.device)
                            #img_tensor = img_tensor.to(args.device)
                        s_imgs = torch.stack(st_imgs).to(args.device)
                        style_images = torch.cat((style_images, s_imgs)) if style_images is not None else s_imgs
                        
                        grid_imgs = torch.stack(grid_imgs).to(args.device)
                        
                        
                        style_images = style_images.to(args.device)
                        
                    
                    #save style images
                    style_images = style_images.reshape(-1, 3, 64, 256)
                    style_features = style_extractor(style_images).to(args.device)
                    # style_features = torch.stack(style_featur, dim=0) #We get [320, 5, 2048]
                    #print('style features', style_features.shape)
                    #style_features = style_features.reshape(n, -1).to(args.device)
            else:
                style_images = None
                style_features = None            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                if cor_im == True:
                    x_noise = torch.randn(cor_images.shape).to(args.device)
                
                    timesteps = torch.full((cor_images.shape[0],), 999, device=args.device, dtype=torch.long)
                    
                    noisy_images = noise_scheduler.add_noise(
                        cor_images, x_noise, timesteps
                    )
                    x = noisy_images
                 
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            #scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample

            
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae_model.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x



def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    vae_model = vae.module if hasattr(vae, "module") else vae
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)
        style_feat = []
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            
            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    # word_embedding = label_padding(trans, letter2index, num_tokens, tokens["PAD_TOKEN"], OUTPUT_MAX_LEN)
                    word_embedding = label_padding(trans, num_tokens)
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long() 
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                # text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=args.text_max_length).to(args.device)
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            if style_extractor is not None:
                reshaped_images = style_images.reshape(-1, 3, IMG_HEIGHT, IMG_WIDTH)
                style_features = style_extractor(reshaped_images)
                
            else:
                style_features = None

            if args.latent == True:
                images = vae_model.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
            
            noise = torch.randn(images.shape).to(images.device)
            # Sample a random timestep for each image
            num_train_timesteps = diffusion.noise_steps
            
            timesteps = torch.randint(
                0, num_train_timesteps,
                (images.shape[0],), device=images.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                images, noise, timesteps
            )
            x_t = noisy_images
            t = timesteps
            
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(x_t, timesteps=t, context=text_features, y=s_id, style_extractor=style_features)
            
            loss = mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            ema.step_ema(ema_model, model)

            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)
            
            if lr_scheduler is not None:
                lr_scheduler.step()
    
        if epoch % 10 == 0:
            labels = torch.arange(16).long().to(args.device)
            n=len(labels)
        
            if args.sampling_word == True:
                #generates the word "text" in 16 different styles
                words = ['text']
                for x_text in words: 
                    ema_sampled_images = diffusion.sample(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args)
                    
                    epoch_n = epoch 
                    sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch_n}_ema.jpg"), args)
            else:
                #generates a batch of words
                ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, vae, n=n, x_text=None, labels=labels, args=args, style_extractor=style_extractor, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                epoch_n = epoch 
                sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)
        
            if args.wandb_log==True:
                wandb_sampled_ema= wandb.Image(sampled_ema, caption=f"{x_text}_{epoch}")
                wandb.log({f"Sampled images": wandb_sampled_ema})
            
            torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--model_name', type=str, default='diffusionpen', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(IMG_HEIGHT, IMG_WIDTH))  
    parser.add_argument('--dataset', type=str, default='khmer', help='iam, gnhk, khmer') 
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path') 
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
    parser.add_argument('--sample_text', type=str, default='ពូ', help='single word to generate in sampling mode')
    parser.add_argument('--sample_writer_id', type=int, default=1, help='writer id to use for style images in khmer sampling mode')
    # parser.add_argument('--text_max_length', type=int, default=40)
    
    args = parser.parse_args()

    if args.dataset == 'khmer' and args.save_path == './diffusionpen_iam_model_path':
        args.save_path = './diffusionpen_khmer'

    device = resolve_device(args.device)
    args.device = str(device)
    
    print('torch version', torch.__version__)
    
    if args.wandb_log==True:
        runs = wandb.init(project='DiffusionPen', entity='name_entity', name=args.dataset, config=args)

        wandb.config.update(args)
    
    #create save directories
    setup_logging(args)

    ############################ DATASET ############################
    transform = transforms.Compose([
                        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])
    
    if args.dataset == 'iam':
        print('loading IAM')
        iam_folder = './iam_data/words'
        myDataset = IAMDataset
        style_classes = 339
        if args.level == 'word':
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        else:
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            test_data = myDataset(iam_folder, 'test', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        print('train data', len(train_data))
        
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        
    elif args.dataset == 'gnhk':
        print('loading GNHK')
        myDataset = GNHK_Dataset
        dataset_folder = 'path/to/GNHK'
        style_classes = 515
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        
    elif args.dataset == 'khmer':
        print('loading Khmer CSV')
        dataset_root = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(dataset_root, 'image_processing', 'dataset.csv')
        full_dataset = KhmerDiffusionDataset(csv_path, dataset_root, transforms=transform)
        style_classes = full_dataset.wclasses

        train_idx, val_idx = split_indices_by_writer(full_dataset.rows, val_ratio=0.2, seed=42)
        train_data = torch.utils.data.Subset(full_dataset, train_idx)
        test_data = torch.utils.data.Subset(full_dataset, val_idx)

        character_classes = build_character_classes_from_csv(csv_path) + [' ']
        # character_classes = build_character_classes_from_csv(csv_path)
        # if ' ' not in character_classes:
        #     character_classes.append(' ')
        #
        # global char_classes, letter2index, index2letter, tokens, num_tokens
        # char_classes, letter2index, index2letter = build_label_dictionary(character_classes)
        # if not tok:
        #     tokens = {"PAD_TOKEN": char_classes}
        # else:
        #     tokens = {"GO_TOKEN": char_classes, "END_TOKEN": char_classes + 1, "PAD_TOKEN": char_classes + 2}
        # num_tokens = len(tokens.keys())

    # if args.dataset == 'khmer' and args.text_max_length == 40:
    #     args.text_max_length = 200

    num_workers = int(os.environ.get('DATALOADER_WORKERS', str(args.num_workers)))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    if args.dataset != 'khmer':
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    
    ######################### MODEL #######################################
    if args.model_name == 'wordstylist':
        vocab_size = len(character_classes) + 2
        print('vocab size', vocab_size)
    else:
        vocab_size = len(character_classes)
    print('Vocab size: ', vocab_size)
    
    if str(device).startswith('mps'):
        device_ids = [0]
        args.dataparallel = False
    elif args.dataparallel==True:
        device_ids = [3,4]
        print('using dataparallel with device:', device_ids)
    else:
        if str(device).startswith('cuda'):
            idx = int(''.join(filter(str.isdigit, str(device))))
            device_ids = [idx]
        else:
            device_ids = []
    #unet = unet.to(args.device)

    if args.model_name == 'diffusionpen' or args.model_name == 'akharapen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        if args.dataparallel:
            text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
        
    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None
    
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)#.to(args.device)
    
    if args.dataparallel:
        unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)
    
    #print('unet parameters')
    #print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None 

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    #load from last checkpoint
    
    if args.load_check==True:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Loaded models and optimizer')
    
    if args.latent==True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        if args.dataparallel:
            vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    #add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    #### STYLE ####
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    PATH = args.style_path 
    
    state_dict = torch.load(PATH, map_location=args.device)
    model_dict = feature_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    if args.dataparallel:
        feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)
    
    elif args.train_mode == 'sampling':
        
        print('Sampling started....')
        
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        print('unet loaded')
        unet.eval()
        
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        ema_model.eval()

        os.makedirs('./image_samples', exist_ok=True)

        if args.dataset == 'khmer':
            sample_text = args.sample_text
            sample_writer_id = args.sample_writer_id
            sample_writer_label = full_dataset.writer_id_to_index[int(sample_writer_id)]
            labels = torch.tensor([sample_writer_label]).long().to(args.device)
            style_images_override = build_style_images_for_writer(full_dataset, sample_writer_id, count=5)

            ema_sampled_images = diffusion.sampling(
                ema_model,
                vae,
                n=len(labels),
                x_text=sample_text,
                labels=labels,
                args=args,
                style_extractor=feature_extractor,
                noise_scheduler=ddim,
                transform=transform,
                character_classes=None,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                run_idx=None,
                style_images_override=style_images_override,
            )

            output_path = os.path.join('./image_samples', f'{sample_text}_writer_{sample_writer_id}.png')
            save_single_images(ema_sampled_images, output_path, args)
            print(f'Saved sample to {output_path}')
            return
        
        if args.sampling_mode == 'single_sampling':
            x_text = ['text', 'word']
            for x_text in x_text:
                print('Word:', x_text)
                s = random.randint(0, 339) #index for style class
                
                print('style', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)  
                save_single_images(ema_sampled_images, os.path.join(f'./image_samples/', f'{x_text}_style_{s}.png'), args)

        
        elif args.sampling_mode == 'paragraph':
            print('Sampling paragraph')
            #make the code to generate lines
            lines = 'In this work , we focus on style variation . We present a novel method to control the style of the text . Our method is able to mimic various writing styles .'
            fakes= []
            gap = np.ones((64, 16))
            max_line_width = 900
            total_char_count = 0
            avg_char_width = 0
            current_line_width = 0
            longest_word_length = max(len(word) for word in lines.strip().split(' '))
            #print('longest_word_length', longest_word_length)
            #s = random.randint(0, 339)#.long().to(args.device)
            #s = random.randint(0, 161)#.long().to(args.device)
            s = 12 #25 #129 #201
            for word in lines.strip().split(' '):
                print('Word:', word)
                print('Style:', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=word, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, clip_model=None, run_idx=None)  
                #print('ema_sampled_images', ema_sampled_images.shape)
                image = ema_sampled_images.squeeze(0)
                
                im = torchvision.transforms.ToPILImage()(image)
                #reshape to height 32
                im = im.convert("L")
                #save im
                
                #if len(word) < 4:
                    
                im = crop_whitespace_width(im)
                
                im = Image.fromarray(im)
                if len(word) == longest_word_length:
                    max_word_length_width = im.width
                    print('max_word_length_width', max_word_length_width)
                #im.save(f'./_REBUTTAL/{word}.png')
                # Calculate aspect ratio
                aspect_ratio = im.width / im.height
                
                im = np.array(im)
                #im = np.array(resized_img)
                
                fakes.append(im)
            
            # Calculate the scaling factor based on the longest word
            #find the average character width of the max length word
            
            avg_char_width = max_word_length_width / longest_word_length
            print('avg_char_width', avg_char_width)
            #scaling_factor = avg_char_width / (32 * aspect_ratio)  # Aspect ratio of an average character

            # Scale and pad each word
            scaled_padded_words = []
            max_height = 64  # Defined max height for all images
            
            for word, img in zip(lines.strip().split(' '), fakes):
                
                img_pil = Image.fromarray(img)
                as_ratio = img_pil.width / img_pil.height
                #scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
                scaled_width = int(avg_char_width * len(word))
                
                scaled_img = img_pil.resize((scaled_width, int(scaled_width / as_ratio)))
                print(f'Word {word} - scaled_img {scaled_img.size}')
                # Padding
                #if word is in punctuation:
                if word in punctuation:
                    #rescale to height 10
                    w_punc = scaled_img.width
                    h_punc = scaled_img.height
                    as_ratio_punct = w_punc / h_punc
                    if word == '.':
                        scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
                    else:
                        scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
                    #pad on top and leave the image in the bottom
                    padding_bottom = 10
                    padding_top = max_height - scaled_img.height - padding_bottom# All padding goes on top
                      # No padding at the bottom

                    # Apply padding
                    padded_img = np.pad(scaled_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=255)
                else:
                    if scaled_img.height < max_height:
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                    else:
                        #resize to max height while maintaining aspect ratio
                        #ar = scaled_img.width / scaled_img.height
                        
                        scaled_img = scaled_img.resize((int(max_height * as_ratio) - 4, max_height - 4))
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                        
                    #padded_img = np.array(scaled_img)
                #print('padded_img', padded_img.shape)
                scaled_padded_words.append(padded_img)

            # Create a gap array (white space)
            height = 64  # Fixed height for all images
            gap = np.ones((height, 16), dtype=np.uint8) * 255  # White gap

            # Concatenate images with gaps
            sentence_img = gap  # Start with a gap
            lines = [] 
            line_img = gap
            # Concatenate images with gaps
            '''
            sentence_img = gap  # Start with a gap
            for img in scaled_padded_words:
                #print('img', img.shape)
                sentence_img = np.concatenate((sentence_img, img, gap), axis=1)
            '''
            
            for img in scaled_padded_words:
                img_width = img.shape[1] + gap.shape[1]

                if current_line_width + img_width < max_line_width:
                    # Add the image to the current line
                    if line_img.shape[0] == 0:
                        line_img = np.ones((height, 0), dtype=np.uint8) * 255  # Start a new line
                    line_img = np.concatenate((line_img, img, gap), axis=1)
                    current_line_width += img_width #+ gap.shape[1]
                    #print('current_line_width if', current_line_width)
                    # Check if adding this image exceeds the max line width
                else:
                    # Pad the current line with white space to max_line_width
                    remaining_width = max_line_width - current_line_width
                    line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                    lines.append(line_img)

                    # Start a new line with the current word
                    line_img = np.concatenate((gap, img, gap), axis=1)
                    current_line_width = img_width #+ 2 * gap.shape[1]
                    #print('current_line_width else', current_line_width)
            # Add the last line to the lines list
            if current_line_width > 0:
                # Pad the last line to max_line_width
                remaining_width = max_line_width - current_line_width
                line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                lines.append(line_img)
                
            # # Concatenate all lines to form a paragraph, pad them if necessary
            # max_height = max([line.shape[0] for line in lines])
            # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
            # for line in lines:
            #     if line.shape[0] < max_height:
            #         padding = (max_height - line.shape[0]) // 2
            #         line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)
                
            #     #print the shapes
            #     print('line shape', line.shape)
            #print('paragraph shape', paragraph_img.shape)
            paragraph_img = np.concatenate((lines), axis=0)

                
            paragraph_image = Image.fromarray(paragraph_img)
            paragraph_image = paragraph_image.convert("L")    
            
            paragraph_image.save(f'paragraph_style_{s}.png')

    
if __name__ == "__main__":
    main()
  
  


#['ក', 'ខ', 'គ', 'ឃ', 'ង', 'ច', 'ឆ', 'ជ', 'ឈ', 'ញ', 'ដ', 'ឌ', 'ណ', 'ត', 'ថ', 'ទ', 'ធ', 'ន', 'ប', 'ផ', 'ព', 'ភ', 'ម', 'យ', 'រ', 'ល', 'វ', 'ស', 'ហ', 'ឡ', 'អ', 'ឧ', 'ឪ', 'ឯ', 'ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ', 'ំ', 'ះ', 'ៈ', '៉', '៊', '់', '៍', '័', '្', '\u200b', ' ']