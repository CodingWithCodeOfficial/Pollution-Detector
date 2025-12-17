
# Earth Search STAC -> preview images -> fast TensorFlow classifier (+ visuals)
# Requirements: tensorflow, numpy, matplotlib (no extra installs)
# Includes: robust Grad-CAM, EarlyStopping, data augmentation, CSVLogger, auto-saved plots + final gallery

import os, json, urllib.request, urllib.error, random, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# 0) Config (edit to your area)
# -----------------------------
API_SEARCH = "https://earth-search.aws.element84.com/v1/search"
PRIMARY_COLLECTIONS = ["sentinel-2-l2a"]   # first choice: Sentinel-2 L2A
FALLBACK_COLLECTIONS = ["naip"]            # fallback if no decodable previews

# Milton, GA vicinity (edit these)
BBOX = [-84.6, 33.7, -84.2, 34.1]          # xmin, ymin, xmax, ymax (lon/lat WGS84)
DATE_RANGE = "2024-06-01T00:00:00Z/2024-12-01T23:59:59Z"  # RFC-3339 interval
PAGE_LIMIT = 100                           # per-page limit (STAC servers often cap at 100)

IMG_SIZE = (96, 96)                        # small & fast; raise (e.g., 128) for accuracy
BATCH_SIZE = 64
EPOCHS = 100

# Plots dir
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def save_plot(filename):
    """Save current Matplotlib figure to plots/ without closing."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print("Saved:", path)

# -----------------------------
# 1) STAC search with pagination (POST /v1/search per STAC API)
# -----------------------------
def stac_search_paginated(api_url, collections, bbox, datetime, page_limit=100, max_pages=10):
    features = []
    body = {
        "collections": collections,
        "bbox": bbox,
        "datetime": datetime,
        "limit": int(page_limit)
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type":"application/json"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            page = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "<no body>"
        raise RuntimeError(f"STAC search HTTP {e.code}: {e.reason}\nServer says: {err_body}")

    features.extend(page.get("features", []))

    next_href = None
    for link in page.get("links", []):
        if link.get("rel") == "next":
            next_href = link.get("href"); break

    pages_fetched = 1
    while next_href and pages_fetched < max_pages:
        try:
            with urllib.request.urlopen(next_href, timeout=30) as resp:
                page = json.loads(resp.read().decode("utf-8"))
                features.extend(page.get("features", []))
                next_href = None
                for link in page.get("links", []):
                    if link.get("rel") == "next":
                        next_href = link.get("href"); break
                pages_fetched += 1
        except Exception:
            break

    return features

def collect_preview_assets(feature):
    """Return PNG/JPEG preview asset URLs from a STAC feature."""
    urls = []
    assets = feature.get("assets", {})
    for key, meta in assets.items():
        href = meta.get("href", "")
        typ  = meta.get("type", "")
        roles = meta.get("roles", []) or []
        is_image = isinstance(typ, str) and (typ.startswith("image/png") or typ.startswith("image/jpeg"))
        likely_preview = any(r in roles for r in ["thumbnail","overview","visual","quicklook","browse"]) \
                         or key.lower() in ("thumbnail","overview","visual","quicklook","browse")
        if is_image and likely_preview and href.startswith("http"):
            urls.append(href)
    return urls

def fetch_bytes(url, max_retries=2):
    for attempt in range(max_retries+1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except Exception:
            if attempt < max_retries: time.sleep(0.75)
            else: return None

def decode_image_to_uint8(img_bytes):
    # Try JPEG then PNG -> resize -> uint8
    try:
        img = tf.io.decode_jpeg(img_bytes, channels=3)
    except Exception:
        img = tf.io.decode_png(img_bytes, channels=3)
    return tf.image.resize(img, IMG_SIZE, method="bilinear").numpy().astype(np.uint8)

def build_preview_dataset(collections):
    print(f"Querying Earth Search STAC for collections={collections} …")
    features = stac_search_paginated(API_SEARCH, collections, BBOX, DATE_RANGE, page_limit=PAGE_LIMIT, max_pages=10)
    print(f"Items fetched: {len(features)}")
    imgs = []
    for feat in features:
        urls = collect_preview_assets(feat)
        random.shuffle(urls)
        for u in urls[:1]:  # use at most one preview per item
            b = fetch_bytes(u)
            if b is None: continue
            try:
                arr = decode_image_to_uint8(b)
            except Exception:
                continue
            imgs.append(arr)
    if len(imgs) == 0:
        raise RuntimeError("No decodable PNG/JPEG preview assets found in these collections.")
    X = np.stack(imgs, axis=0)
    print("Preview images collected:", X.shape)
    return X

# Try primary, fall back to NAIP if no previews are found
try:
    X_all = build_preview_dataset(PRIMARY_COLLECTIONS)
except Exception as e:
    print("Primary collection failed or yielded no previews:", e)
    print("Falling back to NAIP previews (same BBOX/times)…")
    X_all = build_preview_dataset(FALLBACK_COLLECTIONS)

# -----------------------------
# 2) Weak labels: haze proxy
# -----------------------------
def haze_proxy_labels(X_uint8):
    x = tf.convert_to_tensor(X_uint8, dtype=tf.float32) / 255.0
    gray = tf.image.rgb_to_grayscale(x)
    sob = tf.image.sobel_edges(gray)
    gx, gy = sob[..., 0], sob[..., 1]
    mag = tf.sqrt(tf.square(gx) + tf.square(gy))
    density = tf.reduce_mean(mag, axis=[1,2,3]).numpy()
    thresh = np.median(density)
    y = (density < thresh).astype(np.int64)  # hazier -> pollution_like (1)
    return y, density, thresh

y_all, edge_density, thresh = haze_proxy_labels(X_all)
class_names = ["clear", "pollution_like"]

rng = np.random.default_rng(42)
idx = np.arange(len(X_all)); rng.shuffle(idx)
n = len(idx); n_train = int(0.7*n); n_val = int(0.15*n)
train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
x_train, y_train = X_all[train_idx], y_all[train_idx]
x_val,   y_val   = X_all[val_idx],   y_all[val_idx]
x_test,  y_test  = X_all[test_idx],  y_all[test_idx]

# -----------------------------
# 3) Fast CNN (+ augmentation)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
], name="augmentation")

def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,                        # augmentation first
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),             # light regularization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# -----------------------------
# 4) Train (with EarlyStopping & CSVLogger)
# -----------------------------
def make_tf_ds(x, y, batch=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle: ds = ds.shuffle(len(x), seed=42)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

early = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=5, restore_best_weights=True
)
csvlog = tf.keras.callbacks.CSVLogger(os.path.join(PLOTS_DIR, "training_log.csv"))

history = model.fit(
    make_tf_ds(x_train, y_train),
    epochs=EPOCHS,
    validation_data= make_tf_ds(x_val, y_val, shuffle=False),
    verbose=2,
    callbacks=[early, csvlog]
)

# -----------------------------
# 5) Evaluate & training curves
# -----------------------------
test_loss, test_acc = model.evaluate(make_tf_ds(x_test, y_test, shuffle=False), verbose=0)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, linestyle='--', alpha=0.5); plt.legend()
plt.tight_layout()
save_plot("training_curves.png")
plt.show(); plt.close()

# -----------------------------
# 6) Confusion matrix
# -----------------------------
y_pred_probs = model.predict(x_test, batch_size=128, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=2).numpy()
def plot_confusion_matrix(cm, classes, normalize=False, cmap='Blues', filename=None):
    if normalize:
        cm = cm.astype('float')
        rowsum = cm.sum(axis=1, keepdims=True); rowsum[rowsum==0] = 1
        cm = cm / rowsum
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=0); plt.yticks(ticks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j], fmt),
                     ha='center', va='center',
                     color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    if filename: save_plot(filename)
    plt.show(); plt.close()

plot_confusion_matrix(cm, class_names, normalize=False, filename="confusion_matrix_raw.png")
plot_confusion_matrix(cm, class_names, normalize=True, filename="confusion_matrix_normalized.png")

# -----------------------------
# 7) Visual overlays (top-K, gallery, montage)
# -----------------------------
def show_images_with_topk(x, y_true, y_prob, class_names, rows=3, cols=5, k=2, filename=None):
    n = rows*cols
    idxs = np.random.choice(len(x), n, replace=False)
    plt.figure(figsize=(14,10))
    for i, idx in enumerate(idxs):
        img = x[idx].astype(np.uint8)
        true_label = class_names[y_true[idx]]
        probs = y_prob[idx]
        topk_idx = np.argsort(probs)[-k:][::-1]
        topk_labels = [class_names[i_] for i_ in topk_idx]
        topk_scores = [probs[i_] for i_ in topk_idx]
        overlay_text = "\n".join([f"{l}: {s*100:.1f}%" for l,s in zip(topk_labels, topk_scores)])
        pred_top1 = topk_labels[0]; correct = (pred_top1 == true_label)
        ax = plt.subplot(rows, cols, i+1); ax.imshow(img); ax.axis('off')
        ax.set_title(f"Pred: {pred_top1}\nTrue: {true_label}",
                     color='green' if correct else 'red', fontsize=11)
        plt.text(0.02, 0.02, overlay_text, fontsize=9, color='white',
                 transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.6, pad=4))
    plt.suptitle(f"Top-{k} Predictions (Green=Correct, Red=Incorrect)", fontsize=14, y=0.98)
    plt.tight_layout()
    if filename: save_plot(filename)
    plt.show(); plt.close()

def show_misclassifications(x, y_true, y_pred, class_names, max_count=24, filename=None):
    wrong_idxs = np.where(y_true != y_pred)[0]
    if len(wrong_idxs) == 0:
        print("No misclassifications — nice!"); return
    count = min(max_count, len(wrong_idxs))
    chosen = np.random.choice(wrong_idxs, count, replace=False)
    rows = int(np.ceil(count / 6)); cols = min(6, count)
    plt.figure(figsize=(16, 3.5*rows))
    for i, idx in enumerate(chosen):
        ax = plt.subplot(rows, cols, i+1); ax.imshow(x[idx].astype(np.uint8))
        ax.set_title(f"Pred: {class_names[y_pred[idx]]}\nTrue: {class_names[y_true[idx]]}",
                     color='red', fontsize=10); ax.axis('off')
    plt.suptitle("Misclassifications (Wrong Predictions Only)", fontsize=16, y=0.99)
    plt.tight_layout()
    if filename: save_plot(filename)
    plt.show(); plt.close()

def show_per_class_montage(x, y_pred, class_names, per_class=8, filename=None):
    num_classes = len(class_names)
    plt.figure(figsize=(16, 2.5 * num_classes))
    for c in range(num_classes):
        idxs = np.where(y_pred == c)[0]
        if len(idxs)==0: continue
        chosen = np.random.choice(idxs, min(per_class, len(idxs)), replace=False)
        for j, idx in enumerate(chosen):
            ax = plt.subplot(num_classes, per_class, c*per_class + j + 1)
            ax.imshow(x[idx].astype(np.uint8)); ax.axis('off')
            if j==0: ax.set_ylabel(class_names[c], rotation=0, labelpad=20, fontsize=10)
    plt.suptitle("Images by Predicted Class", fontsize=16, y=0.99)
    plt.tight_layout()
    if filename: save_plot(filename)
    plt.show(); plt.close()

show_images_with_topk(x_test, y_test, y_pred_probs, class_names, rows=3, cols=5, k=2, filename="topk_overlays.png")
show_misclassifications(x_test, y_test, y_pred, class_names, max_count=24, filename="misclass_gallery.png")
show_per_class_montage(x_test, y_pred, class_names, per_class=8, filename="per_class_montage.png")

# -----------------------------
# 8) Grad-CAM (robust builder — no reliance on model.input/model.output)
# -----------------------------
def build_grad_cam_model(model, input_shape):
    """
    Build a functional graph that reuses the trained layer objects, returning:
      - last Conv2D feature map
      - final predictions
    """
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    last_conv_output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError("No Conv2D layer found in the model for Grad-CAM.")

    grad_model = tf.keras.Model(inputs=inp, outputs=[last_conv_output, x])
    return grad_model

def make_gradcam_heatmap(img_array, model):
    grad_model = build_grad_cam_model(model, (IMG_SIZE[0], IMG_SIZE[1], 3))
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-9)
    return heatmap.numpy()

def overlay_gradcam_on_image(img_uint8, heatmap, alpha=0.5, cmap='jet'):
    import matplotlib.cm as cm
    h = tf.image.resize(heatmap[..., np.newaxis],
                        (img_uint8.shape[0], img_uint8.shape[1])).numpy().squeeze()
    colored = (cm.get_cmap(cmap)(h)[..., :3] * 255).astype(np.uint8)
    return (alpha * colored + (1 - alpha) * img_uint8).astype(np.uint8)

def show_gradcam_samples(x, y_true, model, class_names, num_samples=8, filename="gradcam_overlays.png"):
    idxs = np.random.choice(len(x), num_samples, replace=False)
    plt.figure(figsize=(14, 8))
    for i, idx in enumerate(idxs):
        img_uint8 = x[idx].astype(np.uint8)
        img_input = img_uint8[np.newaxis, ...].astype(np.float32)  # model has Rescaling(1/255)

        heatmap = make_gradcam_heatmap(img_input, model)
        overlay = overlay_gradcam_on_image(img_uint8, heatmap, alpha=0.5)

        pred = np.argmax(model.predict(img_input, verbose=0)[0])
        ax = plt.subplot(2, num_samples // 2, i + 1)
        ax.imshow(overlay)
        ax.set_title(f"Pred: {class_names[pred]}\nTrue: {class_names[y_true[idx]]}",
                     color='green' if pred == y_true[idx] else 'red', fontsize=10)
        ax.axis('off')

    plt.suptitle("Grad-CAM: Model Attention Overlaid on Image", fontsize=16, y=0.99)
    plt.tight_layout()
    save_plot(filename)
    plt.show(); plt.close()

show_gradcam_samples(x_test, y_test, model, class_names, num_samples=8, filename="gradcam_overlays.png")

# -----------------------------
# 10) Show saved plots (gallery)
# -----------------------------
def show_saved_plots(plot_files, cols=3):
    """Read the PNGs we saved into PLOTS_DIR and show them in a single grid."""
    imgs = []
    for fn in plot_files:
        path = os.path.join(PLOTS_DIR, fn)
        if os.path.exists(path):
            try:
                img = plt.imread(path)
                imgs.append((fn, img))
            except Exception as e:
                print(f"Could not read {path}: {e}")
        else:
            print("Missing:", path)

    if not imgs:
        print("No saved plot images found.")
        return

    rows = int(np.ceil(len(imgs) / cols))
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (name, img) in enumerate(imgs):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name, fontsize=10)
    plt.suptitle("Saved plots", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.show(); plt.close()

## List of plot files produced earlier
plot_files = [
    "training_curves.png",
    "confusion_matrix_raw.png",
    "confusion_matrix_normalized.png",
    "topk_overlays.png",
    "misclass_gallery.png",
    "per_class_montage.png",
    "gradcam_overlays.png",
]

# -----------------------------
# 9) Save model
# -----------------------------
SAVE_PATH = "earthsearch_preview_haze_model.keras"
model.save(SAVE_PATH)
print("Model saved to:", os.path.abspath(SAVE_PATH))

# Show gallery of saved plots