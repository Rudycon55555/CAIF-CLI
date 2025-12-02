# CAIF-CLI.py — Code AI Fast Command Line Interface (Dependency-free, refined)
# Portable: uses only Python standard library modules.
# Features:
# - Robust dataset parsing with explicit schema and clear diagnostics
# - Pure-Python single-hidden-layer neural network with forward/backward passes
# - Supervised and unsupervised (autoencoder) training modes
# - Training feedback: avg loss, gradient clipping, optional input/output normalization
# - EXPORT MODEL writes a standalone dependency-free .py with predict()
# - Clear, predictable CLI parser with strict token handling

import sys
import os
import socket
import math
import random

# ======================================================================
# UTILITIES
# ======================================================================

def _coerce_value(s):
    """Coerce to int, float, bool, or strip quotes and keep as string."""
    v = s.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        pass
    return v

def _split_quoted_csv(s):
    """
    Split a CSV-like string that may contain quoted segments.
    Example: '"a, b", c, "d"' -> ['a, b', 'c', 'd']
    """
    items = []
    buf = []
    in_quote = False
    quote_char = None
    for ch in s:
        if in_quote:
            if ch == quote_char:
                in_quote = False
                quote_char = None
            else:
                buf.append(ch)
        else:
            if ch in ('"', "'"):
                in_quote = True
                quote_char = ch
            elif ch == ',':
                item = ''.join(buf).strip()
                if item:
                    items.append(item)
                buf = []
            else:
                buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        items.append(last)
    return items

def _is_number(tok):
    try:
        float(tok)
        return True
    except Exception:
        return False

# ======================================================================
# DATA LOADER (Explicit schemas, robust parsing)
# ======================================================================

def _parse_numeric_pairs_from_text(text, delimiter=None):
    """
    Parse lines into (inputs, targets) with explicit delimiters:
      - Tab: x1,x2,...,xn \t y1,y2,...,ym
      - Pipe: x1,x2,...,xn | y1,y2,...,ym
      - If delimiter is None, auto-detect per line: prefer tab, then pipe.
    Supports header lines:
      # SCHEMA: input_dim=3, output_dim=2, delimiter="|"
    Returns: list[(list[float], list[float])], meta dict with 'input_dim','output_dim','delimiter'
    Diagnostics printed for malformed lines.
    """
    dataset = []
    meta = {'input_dim': None, 'output_dim': None, 'delimiter': delimiter}

    def parse_schema(line):
        # Example: # SCHEMA: input_dim=3, output_dim=2, delimiter="|"
        if not line.lower().startswith('# schema:'):
            return False
        body = line[len('# SCHEMA:'):].strip()
        parts = [p.strip() for p in body.split(',')]
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                k = k.strip().lower()
                v = _coerce_value(v.strip())
                if k in ('input_dim', 'output_dim'):
                    try:
                        meta[k] = int(v)
                    except Exception:
                        print(f"[WARN] SCHEMA value for {k} must be integer. Got: {v}")
                elif k == 'delimiter':
                    if isinstance(v, str) and v in ('|', '\t'):
                        meta['delimiter'] = v
        return True

    lines = text.splitlines()
    for idx, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            parse_schema(line)
            continue

        # decide delimiter
        d = meta['delimiter']
        if d is None:
            if '\t' in line:
                d = '\t'
            elif '|' in line:
                d = '|'
            else:
                print(f"[DATA WARN] Line {idx} skipped: no recognized delimiter (expected tab or '|').")
                continue

        if d not in ('\t', '|'):
            print(f"[DATA WARN] Line {idx} skipped: unsupported delimiter '{d}'.")
            continue

        try:
            left, right = line.split(d, 1)
        except Exception:
            print(f"[DATA WARN] Line {idx} skipped: missing delimiter '{d}'.")
            continue

        def parse_vec(vec_str):
            tokens = [t.strip() for t in vec_str.replace(';', ',').split(',') if t.strip()]
            if not tokens:
                return None
            if not all(_is_number(t) for t in tokens):
                return None
            return [float(t) for t in tokens]

        xin = parse_vec(left)
        yout = parse_vec(right)
        if xin is None or yout is None:
            print(f"[DATA WARN] Line {idx} skipped: non-numeric or empty vector.")
            continue

        dataset.append((xin, yout))
        # update inferred dims
        if meta['input_dim'] is None:
            meta['input_dim'] = len(xin)
        if meta['output_dim'] is None:
            meta['output_dim'] = len(yout)

    if meta['delimiter'] is None:
        meta['delimiter'] = delimiter or '|'
    return dataset, meta

def _load_data(filepaths, config):
    """IMPORT DATA: read local files or URLs; parse numeric pairs; update config['dataset'] and config['meta']."""
    import urllib.request  # stdlib only
    print(f"[DATA] Reading data from sources: {filepaths}")

    config.setdefault('data_files', [])
    config.setdefault('dataset', [])
    config.setdefault('meta', {'input_dim': None, 'output_dim': None, 'delimiter': None})

    total_parsed = 0
    for source in filepaths:
        try:
            if source.startswith('http://') or source.startswith('https://'):
                with urllib.request.urlopen(source) as response:
                    content = response.read().decode('utf-8', errors='replace')
                    print(f"   -> Fetched {len(content)} bytes from URL: {source}")
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"   -> Read {len(content)} characters from file: {source}")

            parsed, meta = _parse_numeric_pairs_from_text(content, delimiter=config['meta'].get('delimiter'))
            if parsed:
                config['dataset'].extend(parsed)
                # unify meta (prefer explicit values)
                for k in ('input_dim', 'output_dim', 'delimiter'):
                    if config['meta'].get(k) is None and meta.get(k) is not None:
                        config['meta'][k] = meta[k]
                print(f"   -> Parsed {len(parsed)} pairs from '{source}'.")
                total_parsed += len(parsed)
            else:
                print(f"   -> No numeric pairs found in '{source}'.")
            config['data_files'].append(source)
        except Exception as e:
            print(f"[ERROR] Data load failed for '{source}': {e}")

    print(f"[DATA] Total dataset entries: {len(config['dataset'])} (added {total_parsed})")
    # summarize dimensionality
    if config['meta']['input_dim'] and config['meta']['output_dim']:
        print(f"[DATA] Inferred dims: input_dim={config['meta']['input_dim']}, output_dim={config['meta']['output_dim']}")
    return True

# ======================================================================
# PURE-PYTHON NEURAL NETWORK (Single hidden layer, MSE loss, SGD)
# ======================================================================

def _init_weights(input_dim, hidden_dim, output_dim, seed=None):
    """Initialize weights and biases with small random values (pure Python)."""
    if seed is not None:
        random.seed(seed)
    def rand_matrix(rows, cols, scale=0.1):
        return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]
    W1 = rand_matrix(hidden_dim, input_dim)   # hidden_dim x input_dim
    b1 = [0.0 for _ in range(hidden_dim)]
    W2 = rand_matrix(output_dim, hidden_dim)  # output_dim x hidden_dim
    b2 = [0.0 for _ in range(output_dim)]
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def _dot_row_vec(row, vec):
    s = 0.0
    for i in range(len(row)):
        s += row[i] * vec[i]
    return s

def _apply_tanh(v):
    return [math.tanh(x) for x in v]

def _apply_tanh_derivative(v):
    # v is tanh(x) values; derivative = 1 - tanh^2(x)
    return [1.0 - (x * x) for x in v]

def _forward(weights, x):
    """Forward pass: returns hidden_activation (tanh) and output (linear)."""
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    hidden_raw = []
    for i in range(len(W1)):
        hidden_raw.append(_dot_row_vec(W1[i], x) + b1[i])
    hidden = _apply_tanh(hidden_raw)
    out = []
    for i in range(len(W2)):
        out.append(_dot_row_vec(W2[i], hidden) + b2[i])
    return hidden, out

def _mse_loss(pred, target):
    s = 0.0
    n = max(1, len(pred))
    for p, t in zip(pred, target):
        d = p - t
        s += d * d
    return s / n

def _clip(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value

def _backward(weights, x, hidden, pred, target, lr, grad_clip=None):
    """Backward pass: compute gradients and update weights in-place using SGD with optional gradient clipping."""
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    n_out = len(pred)
    d_out = [2.0 * (pred[i] - target[i]) / max(1, n_out) for i in range(n_out)]
    if grad_clip is not None:
        d_out = [_clip(g, -grad_clip, grad_clip) for g in d_out]
    # Gradients for W2 and b2
    for i in range(len(W2)):
        for j in range(len(W2[0])):
            grad = d_out[i] * hidden[j]
            if grad_clip is not None:
                grad = _clip(grad, -grad_clip, grad_clip)
            W2[i][j] -= lr * grad
        b2[i] -= lr * d_out[i]
    # Backprop into hidden
    dh = [0.0 for _ in range(len(hidden))]
    for j in range(len(hidden)):
        s = 0.0
        for i in range(len(W2)):
            s += W2[i][j] * d_out[i]
        dh[j] = s
    tanh_deriv = _apply_tanh_derivative(hidden)
    dh = [dh[j] * tanh_deriv[j] for j in range(len(dh))]
    # Gradients for W1 and b1
    for i in range(len(W1)):
        for j in range(len(W1[0])):
            grad = dh[i] * x[j]
            if grad_clip is not None:
                grad = _clip(grad, -grad_clip, grad_clip)
            W1[i][j] -= lr * grad
        b1[i] -= lr * dh[i]
    # weights updated in-place

# ======================================================================
# HIGH-LEVEL MODEL LIFECYCLE FUNCTIONS
# ======================================================================

def _initialize_model(model_type, parameters, data_meta=None):
    """Initialize model metadata and weights based on parameters and optional data_meta."""
    print(f"[MODEL] Initializing {model_type} structure.")
    # Prefer data_meta dims if not explicitly set
    input_dim = int(parameters.get('input_dim', data_meta.get('input_dim', 1) if data_meta else 1))
    output_dim = int(parameters.get('output_dim', data_meta.get('output_dim', 1) if data_meta else 1))
    hidden_neurons = int(parameters.get('hidden_neurons', 16))
    seed = parameters.get('seed', None)
    weights = _init_weights(input_dim, hidden_neurons, output_dim, seed=seed)
    model = {
        'type': model_type,
        'is_trained': False,
        'context_map': {},
        'weights': weights,
        'input_dim': input_dim,
        'hidden_neurons': hidden_neurons,
        'output_dim': output_dim,
        'norm': {
            'input_mean': None, 'input_std': None,
            'output_mean': None, 'output_std': None,
        }
    }
    print(f"   -> input_dim={input_dim}, hidden_neurons={hidden_neurons}, output_dim={output_dim}")
    return model

def _compute_norm(dataset, input_dim, output_dim):
    """Compute per-dimension mean/std for inputs/targets."""
    if not dataset:
        return None
    def mean_std(vectors, dim):
        sums = [0.0] for_means = [0.0]*dim
        for_means = [0.0]*dim
        for_std = [0.0]*dim
        n = 0
        for vec in vectors:
            v = list(vec[:dim]) + [0.0] * max(0, dim - len(vec))
            for i in range(dim):
                for_means[i] += v[i]
            n += 1
        means = [m / max(1, n) for m in for_means]
        # compute std
        for vec in vectors:
            v = list(vec[:dim]) + [0.0] * max(0, dim - len(vec))
            for i in range(dim):
                d = v[i] - means[i]
                for_std[i] += d * d
        stds = [math.sqrt(s / max(1, n)) or 1.0 for s in for_std]
        return means, stds
    xs = [x for x, _ in dataset]
    ys = [y for _, y in dataset]
    in_mean, in_std = mean_std(xs, input_dim)
    out_mean, out_std = mean_std(ys, output_dim)
    return {'input_mean': in_mean, 'input_std': in_std, 'output_mean': out_mean, 'output_std': out_std}

def _apply_norm(vec, mean, std, dim):
    v = list(vec[:dim]) + [0.0] * max(0, dim - len(vec))
    return [(v[i] - mean[i]) / (std[i] or 1.0) for i in range(dim)]

def _denorm(vec, mean, std, dim):
    v = list(vec[:dim]) + [0.0] * max(0, dim - len(vec))
    return [(v[i] * (std[i] or 1.0)) + mean[i] for i in range(dim)]

def _run_training(model, epochs, mode="supervised", config=None):
    """
    Train the model with pure-Python SGD.
    Modes:
      - supervised: uses provided targets
      - unsupervised: autoencoder (targets = inputs), output_dim must equal input_dim; or will align to min dims
    Parameters (config['parameters']):
      - learning_rate (float)
      - batch_size (int)
      - grad_clip (float or None)
      - normalize (bool) — apply dataset normalization for stable training
      - shuffle (bool) — shuffle per epoch
    """
    if not model:
        raise ValueError("Model must be initialized before training.")
    if config is None:
        config = {}
    dataset = config.get('dataset', [])
    params = config.get('parameters', {})

    if not dataset:
        print("[TRAIN] No dataset available; aborting training.")
        return model

    lr = float(params.get('learning_rate', 0.01))
    batch_size = max(1, int(params.get('batch_size', 1)))
    grad_clip = params.get('grad_clip', None)
    normalize = bool(params.get('normalize', True))
    shuffle = bool(params.get('shuffle', True))

    # Unsupervised mode: use inputs as targets
    if mode.lower() == 'unsupervised':
        # Align output_dim with input_dim if needed
        if model['output_dim'] != model['input_dim']:
            print(f"[TRAIN INFO] Unsupervised mode: aligning output_dim={model['output_dim']} to input_dim={model['input_dim']}.")
            model['output_dim'] = model['input_dim']
            # Re-init output layer to match dims
            hidden_dim = model['hidden_neurons']
            model['weights']['W2'] = _init_weights(model['input_dim'], hidden_dim, model['output_dim'])['W2']
            model['weights']['b2'] = [0.0 for _ in range(model['output_dim'])]

    # Compute normalization stats
    if normalize:
        norm = _compute_norm(dataset, model['input_dim'], model['output_dim'])
        if norm:
            model['norm'] = norm
            print("[TRAIN] Normalization enabled (per-dimension mean/std).")
        else:
            print("[TRAIN WARN] Normalization requested but stats unavailable; proceeding without normalization.")

    print(f"[TRAIN] Starting {mode} training: epochs={epochs}, samples={len(dataset)}, lr={lr}, batch={batch_size}, clip={grad_clip}, normalize={normalize}")

    for epoch in range(1, int(epochs) + 1):
        if shuffle:
            random.shuffle(dataset)
        total_loss = 0.0
        count = 0

        for idx in range(0, len(dataset), batch_size):
            batch = dataset[idx: idx + batch_size]
            for x_raw, y_raw in batch:
                # Build x and y with padding/truncation
                x = list(x_raw)
                if mode.lower() == 'unsupervised':
                    y = list(x_raw)
                else:
                    y = list(y_raw)

                # Pad/truncate to dims
                if len(x) < model['input_dim']:
                    x = x + [0.0] * (model['input_dim'] - len(x))
                else:
                    x = x[:model['input_dim']]
                if len(y) < model['output_dim']:
                    y = y + [0.0] * (model['output_dim'] - len(y))
                else:
                    y = y[:model['output_dim']]

                # Normalize if enabled
                if normalize and model['norm']['input_mean'] is not None:
                    x = _apply_norm(x, model['norm']['input_mean'], model['norm']['input_std'], model['input_dim'])
                if normalize and model['norm']['output_mean'] is not None:
                    y = _apply_norm(y, model['norm']['output_mean'], model['norm']['output_std'], model['output_dim'])

                hidden, pred = _forward(model['weights'], x)
                loss = _mse_loss(pred, y)
                total_loss += loss
                count += 1
                _backward(model['weights'], x, hidden, pred, y, lr, grad_clip=grad_clip)

        avg_loss = total_loss / max(1, count)
        print(f"   -> Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")

    model['is_trained'] = True
    print("[TRAIN] Training complete.")
    return model

def _set_connection(model, target_type, address, context):
    """Store connection details in model context_map."""
    if not model:
        raise ValueError("Model must be initialized before setting connections.")
    model['context_map'][context] = {'type': target_type, 'address': address}
    print(f"[CONNECT] Stored {target_type} connection to '{address}' as '{context}'")
    return model

# ======================================================================
# EXPORT: Standalone .py with learned weights and predict()
# ======================================================================

def _export_model(model, name, parameters):
    """
    Export the trained model into a single dependency-free Python file.
    Includes learned weights, normalization (if computed), and predict(input_vector).
    """
    output_filename = f"{name}.py"
    weights = model.get('weights', {})
    context_map = model.get('context_map', {})
    input_dim = model.get('input_dim', 1)
    hidden_neurons = model.get('hidden_neurons', 1)
    output_dim = model.get('output_dim', 1)
    norm = model.get('norm', {})

    def repr_list(obj):
        return repr(obj)

    with open(output_filename, 'w', encoding='utf-8') as out:
        out.write("# Exported CAIF model (dependency-free)\n")
        out.write("# Contains learned weights, optional normalization, and predict(input_vector).\n\n")
        out.write("import math\n")
        out.write("import socket\n\n")
        out.write(f"INPUT_DIM = {input_dim}\n")
        out.write(f"HIDDEN_NEURONS = {hidden_neurons}\n")
        out.write(f"OUTPUT_DIM = {output_dim}\n\n")
        out.write(f"MODEL_CONTEXT = {repr(context_map)}\n\n")
        out.write("# Learned weights and biases\n")
        out.write(f"W1 = {repr_list(weights.get('W1', []))}\n")
        out.write(f"b1 = {repr_list(weights.get('b1', []))}\n")
        out.write(f"W2 = {repr_list(weights.get('W2', []))}\n")
        out.write(f"b2 = {repr_list(weights.get('b2', []))}\n\n")
        out.write("# Optional normalization (None means disabled)\n")
        out.write(f"NORM = {repr(norm)}\n\n")

        out.write("def _dot_row_vec(row, vec):\n")
        out.write("    s = 0.0\n")
        out.write("    for i in range(len(row)):\n")
        out.write("        s += row[i] * vec[i]\n")
        out.write("    return s\n\n")

        out.write("def _apply_tanh(v):\n")
        out.write("    return [math.tanh(x) for x in v]\n\n")

        out.write("def _apply_norm(vec, mean, std, dim):\n")
        out.write("    if mean is None or std is None:\n")
        out.write("        return vec[:dim] + [0.0] * max(0, dim - len(vec))\n")
        out.write("    v = vec[:dim] + [0.0] * max(0, dim - len(vec))\n")
        out.write("    return [(v[i] - mean[i]) / (std[i] or 1.0) for i in range(dim)]\n\n")

        out.write("def _send_xp_command(command_data, target_address):\n")
        out.write("    try:\n")
        out.write("        host, port_str = target_address.split(':')\n")
        out.write("        port = int(port_str)\n")
        out.write("    except Exception:\n")
        out.write("        print('ERROR: Invalid XP address format (must be host:port).')\n")
        out.write("        return False\n")
        out.write("    try:\n")
        out.write("        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n")
        out.write("        s.connect((host, port))\n")
        out.write("        s.sendall((str(command_data)).encode('utf-8') + b'\\n')\n")
        out.write("        s.close()\n")
        out.write("        return True\n")
        out.write("    except Exception as e:\n")
        out.write("        print(f'XP Connection Failed: {e}')\n")
        out.write("        return False\n\n")

        out.write("def _run_tc_tool(tool_name, arguments):\n")
        out.write("    # USER HOOK: replace with local tool logic if desired\n")
        out.write("    print(f'Tool Call Requested: {tool_name}({arguments})')\n")
        out.write("    return f\"Tool '{tool_name}' executed placeholder logic with args: {arguments}\"\n\n")

        out.write("def predict(input_vector, action_threshold=1.0):\n")
        out.write("    # input_vector list is padded/truncated to INPUT_DIM, with optional normalization\n")
        out.write("    x = list(input_vector)\n")
        out.write("    x = _apply_norm(x, NORM.get('input_mean'), NORM.get('input_std'), INPUT_DIM)\n\n")
        out.write("    # Hidden layer\n")
        out.write("    hidden_raw = []\n")
        out.write("    for i in range(len(W1)):\n")
        out.write("        hidden_raw.append(_dot_row_vec(W1[i], x) + b1[i])\n")
        out.write("    hidden = _apply_tanh(hidden_raw)\n\n")
        out.write("    # Output layer (linear)\n")
        out.write("    out = []\n")
        out.write("    for i in range(len(W2)):\n")
        out.write("        out.append(_dot_row_vec(W2[i], hidden) + b2[i])\n\n")
        out.write("    # Basic action routing (optional): choose based on magnitudes\n")
        out.write("    a = out[0] if len(out) > 0 else 0.0\n")
        out.write("    b = out[1] if len(out) > 1 else 0.0\n")
        out.write("    if a > b and a > action_threshold:\n")
        out.write("        xp_ctx = None\n")
        out.write("        for ctx, info in MODEL_CONTEXT.items():\n")
        out.write("            if (info or {}).get('type') == 'XP':\n")
        out.write("                xp_ctx = info\n")
        out.write("                break\n")
        out.write("        if xp_ctx:\n")
        out.write("            _send_xp_command({'command': 'AUTO', 'payload': out}, xp_ctx.get('address'))\n")
        out.write("            return {'output': out, 'action': 'XP', 'address': xp_ctx.get('address')}\n")
        out.write("        return {'output': out, 'action': None}\n")
        out.write("    if b > a and b > action_threshold:\n")
        out.write("        tc_ctx = None\n")
        out.write("        for ctx, info in MODEL_CONTEXT.items():\n")
        out.write("            if (info or {}).get('type') == 'TC':\n")
        out.write("                tc_ctx = info\n")
        out.write("                break\n")
        out.write("        tool_name = (tc_ctx or {}).get('address', 'default_tool')\n")
        out.write("        res = _run_tc_tool(tool_name, {'output': out})\n")
        out.write("        return {'output': out, 'action': 'TC', 'result': res}\n")
        out.write("    return {'output': out, 'action': None}\n\n")

        out.write("if __name__ == '__main__':\n")
        out.write("    print('\\nRunning exported model...\\n')\n")
        out.write("    sample = [0.0] * INPUT_DIM\n")
        out.write("    print('Prediction:', predict(sample))\n")

    print(f"[SUCCESS] Exported model to {output_filename}")
    return True

# ======================================================================
# COMMAND LINE PARSER (Strict tokens)
# ======================================================================

def execute_caif_file(filepath):
    """Parses and executes CAIF commands line by line."""
    if not os.path.exists(filepath):
        print(f"[ERROR] CAIF file not found at '{filepath}'")
        return

    print("=========================================")
    print(f"  CAIF Executor started for: {filepath}")
    print("=========================================")

    config = {'parameters': {}, 'data_files': [], 'dataset': [], 'meta': {'input_dim': None, 'output_dim': None, 'delimiter': None}}
    current_model = None

    def ensure_model_initialized():
        if current_model is None:
            raise RuntimeError("Model not initialized. Use START MODEL first.")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                if line.startswith('SET PARAMETER'):
                    tail = line[len('SET PARAMETER'):].strip()
                    if '=' not in tail:
                        print(f"[ERROR] Malformed SET PARAMETER on line {line_num}: missing '='")
                        continue
                    key, value = tail.split('=', 1)
                    key = key.strip()
                    value = _coerce_value(value)
                    config['parameters'][key] = value
                    print(f"[PARAM] Set {key} = {value}")

                elif line.startswith('IMPORT DATA'):
                    tail = line[len('IMPORT DATA'):].strip()
                    paths = _split_quoted_csv(tail)
                    if not paths:
                        print(f"[ERROR] IMPORT DATA requires at least one source (line {line_num}).")
                        continue
                    _load_data(paths, config)

                elif line.startswith('START MODEL'):
                    tail = line[len('START MODEL'):].strip()
                    model_type = _coerce_value(tail) or "model"
                    current_model = _initialize_model(model_type, config['parameters'], data_meta=config.get('meta', {}))

                elif line.startswith('TRAIN FOR'):
                    # Syntax: TRAIN FOR <epochs>
                    ensure_model_initialized()
                    tail = line[len('TRAIN FOR'):].strip()
                    parts = tail.split()
                    if not parts or not parts[0].isdigit():
                        print(f"[ERROR] TRAIN FOR requires integer epoch count (line {line_num}).")
                        continue
                    epochs = int(parts[0])
                    current_model = _run_training(
                        current_model, epochs, mode="supervised",
                        config={'dataset': config.get('dataset', []), 'parameters': config.get('parameters', {})}
                    )

                elif line.startswith('START TRAINING'):
                    # Syntax options:
                    # START TRAINING
                    # START TRAINING mode supervised epochs 10
                    # START TRAINING mode unsupervised epochs 20
                    ensure_model_initialized()
                    tail = line[len('START TRAINING'):].strip()
                    tokens = tail.split()
                    mode = "supervised"
                    epochs = int(config.get('parameters', {}).get('epochs', 5))
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i].lower()
                        if tok == 'mode' and i + 1 < len(tokens):
                            mode = tokens[i + 1].lower()
                            i += 2
                        elif tok == 'epochs' and i + 1 < len(tokens):
                            try:
                                epochs = int(tokens[i + 1])
                            except Exception:
                                print(f"[WARN] Invalid epochs value on line {line_num}, defaulting to {epochs}.")
                            i += 2
                        else:
                            i += 1
                    current_model = _run_training(
                        current_model, epochs=epochs, mode=mode,
                        config={'dataset': config.get('dataset', []), 'parameters': config.get('parameters', {})}
                    )

                elif line.startswith('CONNECT TO'):
                    ensure_model_initialized()
                    tail = line[len('CONNECT TO'):].strip()
                    # Expected: CONNECT TO <TYPE> "<address>" as <context>
                    parts = tail.split(None, 2)
                    if len(parts) < 2:
                        print(f"[ERROR] Malformed CONNECT TO on line {line_num}.")
                        continue
                    target_type = parts[0].strip()
                    rest = parts[1] if len(parts) >= 2 else ''
                    remainder = parts[2] if len(parts) == 3 else ''
                    addr = None
                    ctx = 'default'
                    combined = (rest + ' ' + remainder).strip()
                    q_start = None
                    q_end = None
                    q_ch = None
                    for idx, ch in enumerate(combined):
                        if ch in ('"', "'"):
                            q_start = idx
                            q_ch = ch
                            break
                    if q_start is not None:
                        for j in range(q_start + 1, len(combined)):
                            if combined[j] == q_ch:
                                q_end = j
                                break
                        if q_end is not None:
                            addr = combined[q_start + 1:q_end]
                            after = combined[q_end + 1:].strip()
                            if after.lower().startswith('as '):
                                ctx = _coerce_value(after[3:].strip())
                        else:
                            addr = combined
                    if addr is None:
                        addr = rest.strip()
                    current_model = _set_connection(current_model, target_type, addr, ctx)

                elif line.startswith('EXPORT MODEL'):
                    ensure_model_initialized()
                    if not current_model.get('is_trained'):
                        print(f"[ERROR] Cannot EXPORT MODEL before training on line {line_num}.")
                        continue
                    tail = line[len('EXPORT MODEL'):].strip()
                    if tail.lower().startswith('as'):
                        model_name = _coerce_value(tail[2:].strip())
                    else:
                        model_name = _coerce_value(tail) or "ExportedModel"
                    _export_model(current_model, model_name, config.get('parameters', {}))

                else:
                    print(f"[ERROR] Unknown command on line {line_num}: {line}")

            except Exception as e:
                print(f"[FATAL] CRASH on line {line_num} ({line}): {e}")

    print("=========================================")
    print("  CAIF Execution Finished.")
    print("=========================================")

# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        execute_caif_file(sys.argv[1])
    else:
        print("Usage: python CAIF-CLI.py <path_to_caif_file>")
        print("\nExample CAIF script contents:")
        print('''# Example.caif
# SCHEMA: input_dim=2, output_dim=1, delimiter="|"
SET PARAMETER hidden_neurons = 8
SET PARAMETER learning_rate = 0.05
SET PARAMETER batch_size = 1
SET PARAMETER grad_clip = 1.0
SET PARAMETER normalize = true
IMPORT DATA "train_pairs.txt"
START MODEL "regressor"
TRAIN FOR 20
START TRAINING mode unsupervised epochs 10
CONNECT TO XP "127.0.0.1:9000" as robot_arm
EXPORT MODEL as "MyTrainedModel"
''')
