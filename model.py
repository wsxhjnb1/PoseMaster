import os
import cv2
import numpy as np
import mediapipe as mp
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

model_name = "DAMO-NLP-SG/VideoLLaMA3-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# ------------------------------
# 1. Set seeds for reproducibility
# ------------------------------
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------
# 2. Initialize MediaPipe Pose estimator
# ------------------------------
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)

def vedio_explain(video, label):
    question = (
        f"Below is a sequence of frames from a push-up exercise with {label} posture. "
        f"Explain in detail why this posture is {label}. Focus on the alignment, elbow position, "
        f"and back posture."
    )
    conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video, "fps": 1, "max_frames": 128}},
            {"type": "text", "text": question},
        ]
    },
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(response)
    return response

# ------------------------------
# 3. Generate sample image data (simulate still images)
# ------------------------------
number_of_image_samples = 300  # e.g., 300 image samples
sample_image_data = []
for i in range(number_of_image_samples):
    if i % 2 == 0:
        color = (255, 255, 255)  # white image for "correct" posture
        label = 1
        feedback = "Good posture!"
    else:
        color = (50, 50, 50)     # gray image for "incorrect" posture
        label = 0
        feedback = "Incorrect posture!"
    img = np.full((480, 640, 3), color, dtype=np.uint8)
    sample_image_data.append({
        "type": "image",
        "data": img,            
        "text": feedback,
        "label": label
    })

# ------------------------------
# 4. Get video data
# ------------------------------
sample_video_data = []
video_path_correct = "./dataset/train/correct/vedio/pushup"
video_path_incorrect = "./dataset/train/incorrect/vedio/pushup"

for video_name in os.listdir(video_path_correct):
    if not video_name.endswith(".mp4"):
        continue
    video_path = os.path.join(video_path_correct, video_name)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    explaination = vedio_explain(video_path, "correct")
    print(explaination)
    sample_video_data.append({
        "type": "video",
        "data": frames,
        "text": explaination,
        "label": 1,
        "keypoints": np.load(os.path.join(video_path_correct, "correct.npy"))  
    })

for video_name in os.listdir(video_path_incorrect):
    if not video_name.endswith(".mp4"):
        continue
    video_path = os.path.join(video_path_incorrect, video_name)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    explaination = vedio_explain(video_path, "correct")
    print(explaination)
    sample_video_data.append({
        "type": "video",
        "data": frames,
        "text": explaination,
        "label": 0,
        "keypoints": np.load(os.path.join(video_path_incorrect, "incorrect.npy"))
    })

# ------------------------------
# 5. Combine image and video samples into one dataset list
# ------------------------------
# To mix both, you could do:
# dataset_samples = sample_image_data + sample_video_data
# For now, we use only video samples.
dataset_samples = sample_video_data

# ------------------------------
# 6. For each sample, extract pose keypoints.
#    For image samples, run pose estimation.
#    For video samples, if keypoints already exist, we keep them.
# ------------------------------
for sample in dataset_samples:
    if sample["type"] == "image":
        img_rgb = cv2.cvtColor(sample["data"], cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(img_rgb)
        if not results.pose_landmarks:
            keypoints = np.zeros(33 * 3, dtype=np.float32)
        else:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints, dtype=np.float32)
        sample["keypoints"] = keypoints
    else:
        # For video samples, assume keypoints were preloaded.
        # (If they are not present, you could run pose estimation on the middle frame.)
        if "keypoints" not in sample or sample["keypoints"] is None:
            mid_idx = len(sample["data"]) // 2
            img_rgb = cv2.cvtColor(sample["data"][mid_idx], cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(img_rgb)
            if not results.pose_landmarks:
                keypoints = np.zeros(33 * 3, dtype=np.float32)
            else:
                landmarks = results.pose_landmarks.landmark
                keypoints = []
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])
                keypoints = np.array(keypoints, dtype=np.float32)
            sample["keypoints"] = keypoints
        else:
            # If preloaded keypoints have extra dimensions, pick the mid frame and flatten to length 99.
            kp_arr = sample["keypoints"]  # e.g., shape (50,150,66)
            mid_idx = kp_arr.shape[0] // 2
            kp_mid = kp_arr[mid_idx]      # shape could be (150,66)
            kp_flat = kp_mid.flatten()
            # Take the first 99 elements (33 landmarks * 3 values)
            sample["keypoints"] = kp_flat[:99]

# ------------------------------
# 7. Define image transformation (resize to 224x224 and normalize)
# ------------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------------------
# 8. Build vocabulary from feedback texts for tokenization
# ------------------------------
all_tokens = []
for sample in dataset_samples:
    tokens = sample["text"].lower().split()
    all_tokens.extend(tokens)
vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
for token in all_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)
vocab_size = len(vocab)
inv_vocab = {idx: word for word, idx in vocab.items()}

# ------------------------------
# 9. Tokenize feedback texts in each sample
# ------------------------------
max_seq_len = 0
for sample in dataset_samples:
    tokens = sample["text"].lower().split()
    token_ids = [vocab["<bos>"]] + [vocab[t] for t in tokens] + [vocab["<eos>"]]
    sample["token_ids"] = token_ids
    max_seq_len = max(max_seq_len, len(token_ids))
for sample in dataset_samples:
    seq = sample["token_ids"]
    pad_len = max_seq_len - len(seq)
    if pad_len > 0:
        seq = seq + [vocab["<pad>"]] * pad_len
    sample["token_ids"] = seq

# ------------------------------
# 10. Split data into train and test sets (e.g., 80% train, 20% test)
# ------------------------------
split_idx = int(len(dataset_samples) * 0.8)
train_list = dataset_samples[:split_idx]
test_list = dataset_samples[split_idx:]
print(f"Total samples: {len(dataset_samples)}, Train: {len(train_list)}, Test: {len(test_list)}")

# ------------------------------
# 11. Create a custom PyTorch Dataset that handles both image and video samples
# ------------------------------
class MixedPostureDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # For video, pick a representative frame (middle frame); for image, use it directly.
        if sample["type"] == "video":
            frames = sample["data"]
            mid_idx = len(frames) // 2
            img = frames[mid_idx]
        else:
            img = sample["data"]
        img_tensor = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # For keypoints, ensure a 1D vector of length 99
        keypts = torch.tensor(sample["keypoints"], dtype=torch.float32)
        text_tokens = torch.tensor(sample["token_ids"], dtype=torch.long)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img_tensor, keypts, text_tokens, label

train_dataset = MixedPostureDataset(train_list, transform)
test_dataset = MixedPostureDataset(test_list, transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

# Release MediaPipe resources
pose_estimator.close()

# Debug: print one batch shapes
for batch in train_loader:
    img_batch, kp_batch, text_batch, label_batch = batch
    print("Image batch shape:", img_batch.shape)    # (B, 3, 224, 224)
    print("Keypoints batch shape:", kp_batch.shape)   # Expect: (B, 99)
    print("Text tokens shape:", text_batch.shape)     # (B, max_seq_len)
    print("Labels:", label_batch)
    break

# ------------------------------
# Model Implementation
# ------------------------------
class PostureCorrectionModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_classes=2):
        super(PostureCorrectionModel, self).__init__()
        # Use the new API for ResNet18:
        self.image_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.image_encoder.fc = nn.Identity()  # remove classification head
        img_feat_dim = 512  # ResNet18 outputs 512-dim vector

        self.img_project = nn.Linear(img_feat_dim, d_model)
        # Pose keypoints MLP encoder: expects input of size 99
        self.kp_encoder = nn.Sequential(
            nn.Linear(99, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        # Transformer encoder & decoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        # Text embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(500, d_model)  # max sequence length 500
        self.vocab_out = nn.Linear(d_model, vocab_size)
        # Optional classifier for posture correctness
        self.classifier = nn.Linear(d_model, num_classes) if num_classes > 0 else None

    def forward(self, image, keypoints, text_input):
        batch_size = image.size(0)
        # Image branch
        img_feat = self.image_encoder(image)           # (B, 512)
        img_emb = self.img_project(img_feat)             # (B, d_model)
        # Pose branch
        kp_emb = self.kp_encoder(keypoints)              # (B, d_model)
        # Prepare encoder input by stacking along the sequence dimension
        # (B, 2, d_model): first token for image, second for pose
        src_seq = torch.stack([img_emb, kp_emb], dim=1)
        src_positions = torch.arange(0, src_seq.size(1), device=src_seq.device).unsqueeze(0).expand(batch_size, -1)
        src_seq = src_seq + self.pos_embedding(src_positions)
        memory = self.transformer_encoder(src_seq)       # (B, 2, d_model)

        # Prepare target embeddings (batch_first: shape (B, seq_len, d_model))
        tgt_seq = text_input  # (B, seq_len)
        tgt_len = tgt_seq.size(1)
        tgt_embeddings = self.token_embedding(tgt_seq)
        tgt_positions = torch.arange(0, tgt_len, device=tgt_seq.device).unsqueeze(0).expand(batch_size, -1)
        tgt_embeddings = tgt_embeddings + self.pos_embedding(tgt_positions)
        # Create tgt_mask (shape: (tgt_len, tgt_len))
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt_seq.device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        decoder_output = self.transformer_decoder(tgt_embeddings, memory, tgt_mask=tgt_mask)  # (B, seq_len, d_model)
        logits = self.vocab_out(decoder_output)  # (B, seq_len, vocab_size)

        # Classification branch: average image and keypoint embeddings
        class_logits = None
        if self.classifier is not None:
            fused_feat = 0.5 * (img_emb + kp_emb)    # (B, d_model)
            class_logits = self.classifier(fused_feat) # (B, num_classes)
        return logits, class_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PostureCorrectionModel(vocab_size=vocab_size, d_model=128, num_classes=2).to(device)

pad_idx = vocab["<pad>"]
text_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
cls_loss_fn = nn.CrossEntropyLoss() if model.classifier is not None else None
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------
# Training Loop
# ------------------------------
num_epochs = 100
model.train()
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for images, keypoints, token_ids, labels in train_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        input_seq = token_ids[:, :-1]  # exclude last token
        target_seq = token_ids[:, 1:]  # exclude first token
        optimizer.zero_grad()
        logits, class_logits = model(images, keypoints, input_seq)
        logits = logits.permute(1, 0, 2)  # (B, seq_len, vocab_size)
        loss_text = text_loss_fn(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
        loss = loss_text
        if cls_loss_fn is not None:
            loss_cls = cls_loss_fn(class_logits, labels)
            loss = loss + loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{num_epochs} - Training loss: {avg_loss:.4f}")

# ------------------------------
# Evaluation
# ------------------------------
model.eval()
bleu_scores = []
correct_cls = 0
total_cls = 0
smoothing_function = SmoothingFunction().method1
for images, keypoints, token_ids, labels in test_loader:
    images = images.to(device)
    keypoints = keypoints.to(device)
    token_ids = token_ids.to(device)
    labels = labels.to(device)
    generated_tokens = []
    bos_token = torch.tensor([[vocab["<bos>"]]], dtype=torch.long, device=device)
    enc_img_feat = model.image_encoder(images)
    enc_img_emb = model.img_project(enc_img_feat)
    enc_kp_emb = model.kp_encoder(keypoints)
    src_seq = torch.stack([enc_img_emb, enc_kp_emb], dim=1)  # (B, 2, d_model)
    src_positions = torch.arange(0, src_seq.size(1), device=device).unsqueeze(0).expand(images.size(0), -1)
    src_seq = src_seq + model.pos_embedding(src_positions)
    memory = model.transformer_encoder(src_seq)
    cur_input = bos_token
    for _ in range(max_seq_len):
        tgt_emb = model.token_embedding(cur_input)
        tgt_positions = torch.arange(0, cur_input.size(1), device=device).unsqueeze(0).expand(images.size(0), -1)
        tgt_emb = tgt_emb + model.pos_embedding(tgt_positions)
        tgt_mask = torch.triu(torch.ones(cur_input.size(1), cur_input.size(1), device=device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        dec_out = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out_logits = model.vocab_out(dec_out)
        next_token = int(torch.argmax(out_logits[:, -1, :][0]))
        if next_token == vocab["<eos>"]:
            break
        generated_tokens.append(next_token)
        cur_input = torch.cat([cur_input, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
    generated_words = [inv_vocab[idx] for idx in generated_tokens if idx not in (vocab["<bos>"], vocab["<eos>"], vocab["<pad>"])]
    reference_tokens = [inv_vocab[idx] for idx in token_ids[0].tolist() if idx not in (vocab["<bos>"], vocab["<eos>"], vocab["<pad>"])]
    bleu = sentence_bleu([reference_tokens], generated_words, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu_scores.append(bleu)
    if model.classifier is not None:
        pred_cls = torch.argmax(model.classifier(0.5*(enc_img_emb + enc_kp_emb)), dim=1)
        correct_cls += int(pred_cls.item() == labels.item())
        total_cls += 1
avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
cls_acc = (correct_cls / total_cls) * 100 if total_cls > 0 else 0.0
print(f"Average BLEU score on test set: {avg_bleu:.3f}")
if total_cls > 0:
    print(f"Classification Accuracy on test set: {cls_acc:.2f}%")

# ------------------------------
# Convert to TorchScript for deployment
# ------------------------------
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("posture_correction_model.pt")
print("Model saved as TorchScript to posture_correction_model.pt")

# ------------------------------
# Inference on a new image with GPU processing and square resizing for pose estimation
# ------------------------------
test_img_path = "good.jpg"
image_bgr = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = image_rgb.shape
square_dim = max(h, w)
square_img = cv2.resize(image_rgb, (square_dim, square_dim))

with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
    results = pose.process(square_img)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints = [0.0] * (33 * 3)
keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

img_tensor = transform(image_rgb).unsqueeze(0).to(device)

with torch.no_grad():
    enc_img_feat = model.image_encoder(img_tensor)
    enc_img_emb = model.img_project(enc_img_feat)
    enc_kp_emb = model.kp_encoder(keypoints)
    src_seq = torch.stack([enc_img_emb, enc_kp_emb], dim=1)
    src_seq = src_seq + model.pos_embedding(torch.arange(0, src_seq.size(1), device=device).unsqueeze(0).expand(img_tensor.size(0), -1))
    memory = model.transformer_encoder(src_seq)
    generated_tokens = []
    cur_input = torch.tensor([[vocab["<bos>"]]], dtype=torch.long, device=device)
    for _ in range(50):
        tgt_emb = model.token_embedding(cur_input)
        tgt_positions = torch.arange(0, cur_input.size(1), device=device).unsqueeze(0).expand(img_tensor.size(0), -1)
        tgt_emb = tgt_emb + model.pos_embedding(tgt_positions)
        tgt_mask = torch.triu(torch.ones(cur_input.size(1), cur_input.size(1), device=device), diagonal=1)
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        dec_out = model.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        out_logits = model.vocab_out(dec_out)
        next_token = int(torch.argmax(out_logits[:, -1, :][0]))
        if next_token == vocab["<eos>"]:
            break
        generated_tokens.append(next_token)
        cur_input = torch.cat([cur_input, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)
    output_words = [inv_vocab[idx] for idx in generated_tokens if idx not in (vocab["<bos>"], vocab["<eos>"], vocab["<pad>"])]
    feedback_text = " ".join(output_words)
    
    posture_label = None
    if hasattr(model, 'classifier') and model.classifier is not None:
        class_logits = model.classifier(0.5*(enc_img_emb + enc_kp_emb))
        class_idx = int(torch.argmax(class_logits, dim=1).item())
        posture_label = "correct" if class_idx == 1 else "incorrect"

print("Generated Feedback:", feedback_text)
if posture_label:
    print("Posture Classification:", posture_label)